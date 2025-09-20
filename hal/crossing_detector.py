"""
Intersection detection using the bentley_ottmann library.
"""

from bentley_ottmann.core.base import sweep
from ground.base import get_context
from typing import List, Tuple


class CrossingDetector:
    """
    Intersection detection using the bentley_ottmann library.
    Only contains methods needed for visualization intersection detection.
    """

    def __init__(self):
        """Initialize crossing detector."""
        self.context = get_context()
        self.Point = self.context.point_cls
        self.Segment = self.context.segment_cls

    def _find_intersections_from_segments(self, segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                                          qubit_positions: set = None) -> List[Tuple[float, float]]:
        """
        Find intersections from a list of line segments, excluding qubit positions.

        Args:
            segments: List of line segments ((x1,y1), (x2,y2))
            qubit_positions: Set of (x, y) qubit positions to exclude from intersection detection

        Returns:
            List of (x, y) intersection coordinates
        """
        if len(segments) < 2:
            return []

        # Filter out degenerate segments
        valid_segments = [seg for seg in segments if seg[0] != seg[1]]
        if len(valid_segments) < 2:
            return []

        # Convert to bentley_ottmann Segment objects
        bo_segments = []
        for seg in valid_segments:
            start_point = self.Point(seg[0][0], seg[0][1])
            end_point = self.Point(seg[1][0], seg[1][1])
            bo_segments.append(self.Segment(start_point, end_point))

        # Find intersections using bentley_ottmann sweep line algorithm
        events = list(sweep(bo_segments, context=self.context))

        # Extract intersection points from events with tangents
        intersection_points = []
        seen_points = set()

        for event in events:
            # Events with tangents indicate intersections
            if len(event.tangents) > 1:
                point = event.start
                coord = (float(point.x), float(point.y))

                # Avoid duplicates
                if coord in seen_points:
                    continue
                seen_points.add(coord)

                # Filter out qubit positions (intended connection points)
                if qubit_positions and self._is_qubit_position(coord, qubit_positions):
                    continue

                intersection_points.append(coord)

        return intersection_points

    def _is_qubit_position(self, point: Tuple[float, float], qubit_positions: set) -> bool:
        """
        Check if an intersection point is a qubit position (intended connection point).

        Args:
            point: The intersection point to check
            qubit_positions: Set of (x, y) qubit positions

        Returns:
            True if this point is a qubit position, False otherwise
        """
        x, y = point
        tolerance = 1e-6

        for qx, qy in qubit_positions:
            if abs(x - qx) < tolerance and abs(y - qy) < tolerance:
                return True
        return False