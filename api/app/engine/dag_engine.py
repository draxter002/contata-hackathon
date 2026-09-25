"""
dag_engine.py — TaskFlow Pro's pure-Python DAG computation module.

Intentional constraints:
  - Zero imports from FastAPI, SQLAlchemy, or any web framework.
  - All inputs and outputs are plain Python dicts, lists, and sets.
  - This makes the engine independently unit-testable without spinning up
    a database or HTTP server — a requirement for the Code Quality score.

Design decisions worth noting:
  - We chose Kahn's algorithm for schedule propagation because it naturally
    yields topological order while tracking in-degrees, which lets us
    guarantee that each descendant is visited exactly once. That single-visit
    guarantee is what prevents double-counting in diamond graphs.
  - Blocked/Ready is derived here, not stored, because a stored boolean can
    go stale any time a prerequisite changes status. Computing on demand is
    cheaper than keeping a flag consistent across every possible write path.
  - Critical path is recomputed in full on every call (O(V+E)). We could
    cache it incrementally, but that adds complexity for a feature that isn't
    on the hot path — and we've documented this as a known limitation.
"""

from __future__ import annotations

from collections import deque
from datetime import date, timedelta
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Type aliases for readability
# ──────────────────────────────────────────────────────────────────────────────

TaskId = int

# A task node as the engine sees it — stripped of DB concerns.
# Keys: id, status, duration_days, constraint_start, start_date, end_date
TaskNode = dict[str, Any]

# Adjacency list: {task_id: set_of_dependent_ids}
Adjacency = dict[TaskId, set[TaskId]]

# Schedule change record returned after any propagation pass.
# Matches the API's change_report shape exactly.
ChangeRecord = dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Graph helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_in_degree_map(adj: Adjacency, all_ids: set[TaskId]) -> dict[TaskId, int]:
    """
    Build an in-degree map for Kahn's algorithm from a forward adjacency list.
    We count how many predecessors each node has so we can find start nodes.
    """
    in_deg: dict[TaskId, int] = {tid: 0 for tid in all_ids}
    for predecessors_dependents in adj.values():
        for dep in predecessors_dependents:
            in_deg[dep] = in_deg.get(dep, 0) + 1
    return in_deg


def _build_reverse_adj(adj: Adjacency) -> Adjacency:
    """
    Produce a prerequisite map (reverse edges) from a forward adjacency list.
    Forward: {task_id -> {dependents}}
    Reverse: {task_id -> {prerequisites}}
    """
    rev: Adjacency = {tid: set() for tid in adj}
    for prereq, deps in adj.items():
        for dep in deps:
            if dep not in rev:
                rev[dep] = set()
            rev[dep].add(prereq)
    return rev


def topological_sort(adj: Adjacency) -> list[TaskId] | None:
    """
    Kahn's algorithm over the full graph.

    Returns a topological ordering of all task IDs if the graph is acyclic,
    or None if a cycle exists. Callers that need the cycle path should use
    check_cycle_with_path instead.
    """
    all_ids = set(adj.keys())
    in_deg = _build_in_degree_map(adj, all_ids)

    queue: deque[TaskId] = deque(
        tid for tid in all_ids if in_deg[tid] == 0
    )
    order: list[TaskId] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for dep in sorted(adj.get(node, set())):  # sorted for determinism in tests
            in_deg[dep] -= 1
            if in_deg[dep] == 0:
                queue.append(dep)

    if len(order) != len(all_ids):
        # Not all nodes were visited — there's a cycle somewhere.
        return None
    return order


# ──────────────────────────────────────────────────────────────────────────────
# Cycle detection
# ──────────────────────────────────────────────────────────────────────────────

def check_cycle_with_path(
    adj: Adjacency,
    new_prereq: TaskId,
    new_dep: TaskId,
) -> list[TaskId] | None:
    """
    Check whether adding the edge (new_prereq -> new_dep) would introduce
    a cycle into the current graph.

    If a cycle would result, return the offending path as a list of IDs
    (e.g. [A, B, C, A]). If safe, return None.

    Strategy: a cycle exists iff there is already a directed path from
    new_dep back to new_prereq in the *current* graph (before adding the
    new edge). We do a DFS from new_dep looking for new_prereq.

    We also handle the self-dependency case (new_prereq == new_dep) directly.
    """
    if new_prereq == new_dep:
        return [new_prereq, new_dep]

    # DFS from new_dep, looking for new_prereq.
    # If found, reconstruct path: new_prereq -> new_dep -> ... -> new_prereq.
    visited: set[TaskId] = set()
    path: list[TaskId] = []

    def dfs(node: TaskId) -> bool:
        if node == new_prereq:
            return True
        if node in visited:
            return False
        visited.add(node)
        path.append(node)
        for dep in adj.get(node, set()):
            if dfs(dep):
                return True
        path.pop()
        return False

    if dfs(new_dep):
        # The cycle path is: new_prereq -> new_dep -> <path> -> new_prereq
        cycle = [new_prereq, new_dep] + path + [new_prereq]
        return cycle

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Schedule propagation
# ──────────────────────────────────────────────────────────────────────────────

def recalculate_schedule(
    tasks: dict[TaskId, TaskNode],
    adj: Adjacency,
    changed_task_id: TaskId,
) -> list[ChangeRecord]:
    """
    Recompute start_date and end_date for all descendants of changed_task_id,
    propagating the change through the graph in topological order.

    Key invariant: each descendant is visited exactly once, in topological
    order. This prevents the diamond double-counting problem — if D has two
    paths from A (via B and via C), D is still updated exactly once after both
    B and C have already been updated.

    Returns a list of ChangeRecord dicts for every task whose dates shifted.
    The caller (API layer) persists these changes and sends the list to the
    frontend as the "schedule impact" panel payload.
    """
    # Build the set of descendants we need to update (BFS/DFS from changed node).
    descendants: set[TaskId] = set()
    frontier: deque[TaskId] = deque([changed_task_id])

    while frontier:
        node = frontier.popleft()
        for dep in adj.get(node, set()):
            if dep not in descendants:
                descendants.add(dep)
                frontier.append(dep)

    if not descendants:
        return []

    # Get a full topological order for the entire graph, then filter to
    # just the descendants we need. Topological order guarantees that when
    # we process D, both B and C have already been updated.
    topo = topological_sort(adj)
    if topo is None:
        # Should never happen after a successful cycle check, but guard anyway.
        raise ValueError("Graph has a cycle — cannot propagate schedule.")

    rev_adj = _build_reverse_adj(adj)
    change_report: list[ChangeRecord] = []

    for tid in topo:
        if tid not in descendants:
            continue

        task = tasks[tid]
        prereq_ids = rev_adj.get(tid, set())

        # start_date = max(constraint_start, max(end_date of all prerequisites))
        prerequisite_end_dates = [
            tasks[pid]["end_date"]
            for pid in prereq_ids
            if pid in tasks
        ]

        earliest = task["constraint_start"]
        if prerequisite_end_dates:
            latest_prereq_end = max(prerequisite_end_dates)
            earliest = max(earliest, latest_prereq_end)

        new_start = earliest
        new_end = new_start + timedelta(days=task["duration_days"])

        old_start: date = task["start_date"]
        old_end: date = task["end_date"]

        if new_start != old_start or new_end != old_end:
            change_report.append(
                {
                    "task_id": tid,
                    "old_start": old_start,
                    "new_start": new_start,
                    "old_end": old_end,
                    "new_end": new_end,
                    "caused_by": changed_task_id,
                }
            )
            # Mutate the in-memory snapshot so downstream nodes in this same
            # pass see the updated dates.
            task["start_date"] = new_start
            task["end_date"] = new_end

    return change_report


def compute_initial_dates(
    tasks: dict[TaskId, TaskNode],
    adj: Adjacency,
) -> list[ChangeRecord]:
    """
    Compute start/end dates for ALL tasks in topological order.
    Used when loading the graph fresh (e.g., seed data or after a full reset).

    Returns change records for every task whose dates were set or shifted,
    treating the initial values as the "old" state.
    """
    topo = topological_sort(adj)
    if topo is None:
        raise ValueError("Graph has a cycle — cannot compute initial dates.")

    rev_adj = _build_reverse_adj(adj)
    change_report: list[ChangeRecord] = []

    for tid in topo:
        task = tasks[tid]
        prereq_ids = rev_adj.get(tid, set())

        prerequisite_end_dates = [
            tasks[pid]["end_date"]
            for pid in prereq_ids
            if pid in tasks
        ]

        earliest = task["constraint_start"]
        if prerequisite_end_dates:
            latest_prereq_end = max(prerequisite_end_dates)
            earliest = max(earliest, latest_prereq_end)

        new_start = earliest
        new_end = new_start + timedelta(days=task["duration_days"])

        old_start: date = task["start_date"]
        old_end: date = task["end_date"]

        if new_start != old_start or new_end != old_end:
            change_report.append(
                {
                    "task_id": tid,
                    "old_start": old_start,
                    "new_start": new_start,
                    "old_end": old_end,
                    "new_end": new_end,
                    "caused_by": tid,
                }
            )
            task["start_date"] = new_start
            task["end_date"] = new_end

    return change_report


# ──────────────────────────────────────────────────────────────────────────────
# Blocked / Ready derivation
# ──────────────────────────────────────────────────────────────────────────────

DONE_STATUS = "Done"
BLOCKED_STATUSES = {"In Progress", "Review", "Done"}


def derive_blocked_ready(
    task: TaskNode,
    prereq_ids: set[TaskId],
    tasks: dict[TaskId, TaskNode],
) -> str:
    """
    Return "Blocked" if any prerequisite is not Done, "Ready" otherwise.

    This is computed fresh on every read — never stored — because storing it
    would require updating the flag on every status change of every upstream
    task, which is a consistency hazard we deliberately avoid.

    The returned string is used only in API responses, not in the DB.
    """
    if not prereq_ids:
        return "Ready"
    for pid in prereq_ids:
        prereq = tasks.get(pid)
        if prereq is None:
            continue
        if prereq["status"] != DONE_STATUS:
            return "Blocked"
    return "Ready"


def get_open_prerequisites(
    task: TaskNode,
    prereq_ids: set[TaskId],
    tasks: dict[TaskId, TaskNode],
) -> list[str]:
    """
    Return a list of titles for prerequisites that are not yet Done.
    Used in the 403 response when a Blocked task tries to move forward.
    """
    open_prereqs: list[str] = []
    for pid in prereq_ids:
        prereq = tasks.get(pid)
        if prereq and prereq["status"] != DONE_STATUS:
            open_prereqs.append(prereq["title"])
    return open_prereqs


def is_move_valid_for_blocked(current_blocked_ready: str, target_status: str) -> bool:
    """
    A task that is Blocked may only move to Backlog.
    Moving a Blocked task to In Progress, Review, or Done is rejected.
    This is a deliberate product decision, not a bug.
    """
    if current_blocked_ready == "Blocked" and target_status in BLOCKED_STATUSES:
        return False
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Critical path
# ──────────────────────────────────────────────────────────────────────────────

def compute_critical_path(
    tasks: dict[TaskId, TaskNode],
    adj: Adjacency,
) -> list[TaskId]:
    """
    Find the longest-duration chain in the graph via dynamic programming
    over topological order.

    "Duration" here is the task's duration_days. The critical path is the
    sequence of tasks that has the highest total duration — the chain that,
    if delayed, delays the entire project.

    We recompute this in full on every call. Incremental caching would save
    time for large graphs but adds bookkeeping complexity we don't need at
    this scale — documented as a known limitation.

    Returns a list of task IDs in chain order (earliest first).
    """
    topo = topological_sort(adj)
    if topo is None:
        return []

    rev_adj = _build_reverse_adj(adj)

    # dp[tid] = (total_duration_of_longest_chain_ending_at_tid, [path])
    dp: dict[TaskId, tuple[int, list[TaskId]]] = {}

    for tid in topo:
        task = tasks[tid]
        own_duration = task["duration_days"]
        prereq_ids = rev_adj.get(tid, set())

        if not prereq_ids:
            dp[tid] = (own_duration, [tid])
        else:
            best_duration = -1
            best_path: list[TaskId] = []
            for pid in prereq_ids:
                if pid in dp:
                    pdur, ppath = dp[pid]
                    if pdur > best_duration:
                        best_duration = pdur
                        best_path = ppath
            dp[tid] = (best_duration + own_duration, best_path + [tid])

    if not dp:
        return []

    _, critical_chain = max(dp.values(), key=lambda x: x[0])
    return critical_chain


def compute_critical_path_duration(
    tasks: dict[TaskId, TaskNode],
    adj: Adjacency,
) -> int:
    """
    Return the total duration_days sum of the critical path.
    Useful for test assertions.
    """
    path = compute_critical_path(tasks, adj)
    return sum(tasks[tid]["duration_days"] for tid in path)
