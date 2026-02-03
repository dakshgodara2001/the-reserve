"""Staff assignment using Hungarian algorithm for optimal matching."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class StaffInfo:
    """Staff information for assignment."""

    staff_id: int
    name: str
    x_position: float
    y_position: float
    current_workload: int
    max_workload: int
    skills: List[str]
    is_available: bool
    avg_response_time: float = 0.0


@dataclass
class TaskInfo:
    """Task information for assignment."""

    task_id: str
    table_id: int
    table_x: float
    table_y: float
    priority: float
    required_skills: List[str]
    task_type: str


@dataclass
class Assignment:
    """Assignment result."""

    staff_id: int
    task_id: str
    table_id: int
    cost: float
    reasoning: str


class StaffAssigner:
    """
    Assigns staff to tasks using the Hungarian algorithm.

    Cost matrix considers:
    - Distance to table
    - Current workload
    - Skill match
    - Historical response time
    """

    # Weight factors for cost calculation
    DISTANCE_WEIGHT = 1.0
    WORKLOAD_WEIGHT = 2.0
    SKILL_MISMATCH_PENALTY = 5.0
    RESPONSE_TIME_WEIGHT = 0.5

    # Maximum cost (for impossible assignments)
    MAX_COST = 1000.0

    def __init__(self):
        self.assignment_history: List[Assignment] = []

    def assign_optimal(
        self,
        staff_list: List[StaffInfo],
        tasks: List[TaskInfo],
    ) -> List[Assignment]:
        """
        Find optimal assignment of staff to tasks.

        Uses Hungarian algorithm to minimize total cost.

        Args:
            staff_list: Available staff members
            tasks: Tasks requiring assignment

        Returns:
            List of assignments
        """
        if not staff_list or not tasks:
            return []

        # Filter to available staff
        available_staff = [s for s in staff_list if s.is_available]
        if not available_staff:
            return []

        # Build cost matrix
        cost_matrix = self._build_cost_matrix(available_staff, tasks)

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build assignment results
        assignments = []
        for staff_idx, task_idx in zip(row_ind, col_ind):
            # Skip if cost is too high (impossible assignment)
            if cost_matrix[staff_idx, task_idx] >= self.MAX_COST:
                continue

            # Skip if task index exceeds actual tasks (padding case)
            if task_idx >= len(tasks):
                continue

            staff = available_staff[staff_idx]
            task = tasks[task_idx]

            assignment = Assignment(
                staff_id=staff.staff_id,
                task_id=task.task_id,
                table_id=task.table_id,
                cost=cost_matrix[staff_idx, task_idx],
                reasoning=self._explain_assignment(staff, task, cost_matrix[staff_idx, task_idx]),
            )
            assignments.append(assignment)
            self.assignment_history.append(assignment)

        return assignments

    def _build_cost_matrix(
        self,
        staff_list: List[StaffInfo],
        tasks: List[TaskInfo],
    ) -> np.ndarray:
        """Build cost matrix for staff-task assignment."""
        num_staff = len(staff_list)
        num_tasks = len(tasks)

        # Hungarian algorithm needs square matrix
        size = max(num_staff, num_tasks)
        cost_matrix = np.full((size, size), self.MAX_COST)

        for i, staff in enumerate(staff_list):
            for j, task in enumerate(tasks):
                cost_matrix[i, j] = self._calculate_cost(staff, task)

        return cost_matrix

    def _calculate_cost(self, staff: StaffInfo, task: TaskInfo) -> float:
        """Calculate assignment cost for staff-task pair."""
        # Check availability
        if not staff.is_available:
            return self.MAX_COST

        # Check workload capacity
        if staff.current_workload >= staff.max_workload:
            return self.MAX_COST

        cost = 0.0

        # 1. Distance cost
        distance = self._calculate_distance(
            staff.x_position,
            staff.y_position,
            task.table_x,
            task.table_y,
        )
        cost += distance * self.DISTANCE_WEIGHT

        # 2. Workload cost (prefer less busy staff)
        workload_ratio = staff.current_workload / max(staff.max_workload, 1)
        cost += workload_ratio * self.WORKLOAD_WEIGHT

        # 3. Skill match
        if task.required_skills:
            matched_skills = set(staff.skills) & set(task.required_skills)
            if not matched_skills:
                cost += self.SKILL_MISMATCH_PENALTY
            else:
                # Partial match bonus
                match_ratio = len(matched_skills) / len(task.required_skills)
                cost -= match_ratio * 0.5  # Small bonus for skill match

        # 4. Response time history
        cost += staff.avg_response_time * self.RESPONSE_TIME_WEIGHT

        # 5. Priority boost (reduce cost for high priority tasks)
        priority_factor = task.priority / 100.0  # Normalize to 0-1
        cost *= (1.0 - priority_factor * 0.3)  # Up to 30% reduction for high priority

        return max(0, cost)

    def _calculate_distance(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _explain_assignment(
        self,
        staff: StaffInfo,
        task: TaskInfo,
        cost: float,
    ) -> str:
        """Generate human-readable explanation for assignment."""
        distance = self._calculate_distance(
            staff.x_position,
            staff.y_position,
            task.table_x,
            task.table_y,
        )

        reasons = []

        if distance < 0.2:
            reasons.append("closest to table")
        elif distance < 0.4:
            reasons.append("near table")

        workload_ratio = staff.current_workload / max(staff.max_workload, 1)
        if workload_ratio < 0.5:
            reasons.append("low workload")
        elif workload_ratio < 0.75:
            reasons.append("moderate workload")

        if task.required_skills and set(staff.skills) & set(task.required_skills):
            reasons.append("skill match")

        if not reasons:
            reasons.append("best available")

        return f"{staff.name}: {', '.join(reasons)}"

    def suggest_single(
        self,
        staff_list: List[StaffInfo],
        task: TaskInfo,
    ) -> Optional[Assignment]:
        """
        Suggest best staff member for a single task.

        Simpler than full optimal assignment - just finds lowest cost.
        """
        if not staff_list:
            return None

        best_staff = None
        best_cost = float("inf")

        for staff in staff_list:
            if not staff.is_available:
                continue

            cost = self._calculate_cost(staff, task)
            if cost < best_cost:
                best_cost = cost
                best_staff = staff

        if best_staff is None or best_cost >= self.MAX_COST:
            return None

        return Assignment(
            staff_id=best_staff.staff_id,
            task_id=task.task_id,
            table_id=task.table_id,
            cost=best_cost,
            reasoning=self._explain_assignment(best_staff, task, best_cost),
        )

    def get_staff_recommendations(
        self,
        staff_list: List[StaffInfo],
        task: TaskInfo,
        limit: int = 3,
    ) -> List[Tuple[StaffInfo, float, str]]:
        """
        Get ranked staff recommendations for a task.

        Returns:
            List of (staff, cost, reasoning) tuples
        """
        recommendations = []

        for staff in staff_list:
            cost = self._calculate_cost(staff, task)
            if cost < self.MAX_COST:
                reasoning = self._explain_assignment(staff, task, cost)
                recommendations.append((staff, cost, reasoning))

        recommendations.sort(key=lambda x: x[1])
        return recommendations[:limit]

    def rebalance_workload(
        self,
        staff_list: List[StaffInfo],
        table_assignments: Dict[int, int],  # table_id -> staff_id
    ) -> List[Tuple[int, int, int]]:
        """
        Suggest workload rebalancing.

        Returns:
            List of (table_id, from_staff_id, to_staff_id) suggestions
        """
        suggestions = []

        # Calculate current workloads
        workloads = {}
        for staff in staff_list:
            workloads[staff.staff_id] = len(
                [t for t, s in table_assignments.items() if s == staff.staff_id]
            )

        if not workloads:
            return suggestions

        avg_workload = sum(workloads.values()) / len(workloads)

        # Find overloaded and underloaded staff
        overloaded = [
            sid for sid, wl in workloads.items() if wl > avg_workload + 1
        ]
        underloaded = [
            sid for sid, wl in workloads.items() if wl < avg_workload - 1
        ]

        # Suggest transfers
        for from_staff in overloaded:
            for to_staff in underloaded:
                # Find a table to transfer
                tables_from = [
                    t for t, s in table_assignments.items() if s == from_staff
                ]
                if tables_from:
                    suggestions.append((tables_from[0], from_staff, to_staff))
                    break

        return suggestions

    def get_assignment_stats(self) -> Dict[str, Any]:
        """Get statistics about assignments."""
        if not self.assignment_history:
            return {
                "total_assignments": 0,
                "avg_cost": 0,
                "assignments_by_staff": {},
            }

        costs = [a.cost for a in self.assignment_history]
        by_staff: Dict[int, int] = {}
        for a in self.assignment_history:
            by_staff[a.staff_id] = by_staff.get(a.staff_id, 0) + 1

        return {
            "total_assignments": len(self.assignment_history),
            "avg_cost": sum(costs) / len(costs),
            "min_cost": min(costs),
            "max_cost": max(costs),
            "assignments_by_staff": by_staff,
        }
