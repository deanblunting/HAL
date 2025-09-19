"""
Simplified crossing detection using the line-segment-intersections library.
Much more reliable and efficient than custom implementation.
"""

import linesegmentintersections as lsi
from typing import List, Tuple, Set, Dict, Optional, Union
from collections import defaultdict


class CrossingDetector:
    """
    Simplified crossing detection using the line-segment-intersections library.
    Much more reliable and efficient than custom implementation.
    """

    def __init__(self):
        """Initialize crossing detector."""
        self.layer_segments = defaultdict(list)  # {layer_id: [list of ((x1,y1), (x2,y2)) tuples]}

    def would_create_crossing(self, proposed_path: List[Tuple[int, int]], layer: int) -> bool:
        """
        Check if a proposed path would cross existing paths on the given layer.

        Args:
            proposed_path: List of (x, y) coordinates forming a path
            layer: Layer ID to check on

        Returns:
            True if the path would create a crossing, False otherwise
        """
        return len(self.get_crossing_points(proposed_path, layer)) > 0

    def get_crossing_points(self, proposed_path: List[Tuple[int, int]], layer: int) -> List[Tuple[float, float]]:
        """
        Get intersection points where a proposed path would cross existing paths.

        Args:
            proposed_path: List of (x, y) coordinates forming a path
            layer: Layer ID to check on

        Returns:
            List of (x, y) intersection coordinates where bump transitions should occur
        """
        if len(proposed_path) < 2:
            return []

        # Get existing segments on this layer
        existing_segments = self.layer_segments[layer]
        if not existing_segments:
            return []  # No existing paths to cross

        # Create segments from proposed path
        proposed_segments = []
        for i in range(len(proposed_path) - 1):
            proposed_segments.append((proposed_path[i], proposed_path[i + 1]))

        # Filter out degenerate segments (same start/end point)
        valid_existing = [seg for seg in existing_segments if seg[0] != seg[1]]
        valid_proposed = [seg for seg in proposed_segments if seg[0] != seg[1]]

        if not valid_existing or not valid_proposed:
            return []

        # Combine all segments and use library to check for intersections
        all_segments = valid_existing + valid_proposed

        # Use the library's Bentley-Ottmann algorithm
        intersections = lsi.bentley_ottman(all_segments)

        # Extract coordinates from Intersection objects
        intersection_points = []
        for intersection in intersections:
            try:
                # The intersection object has a point attribute with x, y coordinates
                if hasattr(intersection, 'point'):
                    point = intersection.point
                    if hasattr(point, 'x') and hasattr(point, 'y'):
                        intersection_points.append((float(point.x), float(point.y)))
                    else:
                        # Fallback: try to access as tuple/list
                        intersection_points.append((float(point[0]), float(point[1])))
                elif hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                    # Intersection object directly has x, y attributes
                    intersection_points.append((float(intersection.x), float(intersection.y)))
                else:
                    # Fallback: assume intersection is already a coordinate
                    intersection_points.append((float(intersection[0]), float(intersection[1])))
            except (AttributeError, TypeError, IndexError) as e:
                # Skip malformed intersection objects
                print(f"Warning: Could not extract coordinates from intersection object: {e}")
                continue

        return intersection_points

    def check_segment_crossing(self, p1: Tuple[int, int], p2: Tuple[int, int], layer: int) -> bool:
        """
        Fast check for single segment crossing (optimized for A* pathfinding).

        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            layer: Layer ID to check on

        Returns:
            True if the segment would create a crossing, False otherwise
        """
        return len(self.get_segment_crossing_points(p1, p2, layer)) > 0

    def get_segment_crossing_points(self, p1: Tuple[int, int], p2: Tuple[int, int], layer: int) -> List[Tuple[float, float]]:
        """
        Get intersection points for a single segment crossing.

        Args:
            p1: Start point (x, y)
            p2: End point (x, y)
            layer: Layer ID to check on

        Returns:
            List of (x, y) intersection coordinates where bump transitions should occur
        """
        # Skip crossing detection if points are the same (no segment)
        if p1 == p2:
            return []

        # Get existing segments on this layer
        existing_segments = self.layer_segments[layer]
        if not existing_segments:
            return []

        # Filter out degenerate segments from existing
        valid_existing = [seg for seg in existing_segments if seg[0] != seg[1]]

        if not valid_existing:
            return []

        # Create test segment and combine with existing
        test_segment = (p1, p2)
        all_segments = valid_existing + [test_segment]

        # Use the library to check for intersections
        intersections = lsi.bentley_ottman(all_segments)

        # Extract coordinates from Intersection objects
        intersection_points = []
        for intersection in intersections:
            try:
                # The intersection object has a point attribute with x, y coordinates
                if hasattr(intersection, 'point'):
                    point = intersection.point
                    if hasattr(point, 'x') and hasattr(point, 'y'):
                        intersection_points.append((float(point.x), float(point.y)))
                    else:
                        # Fallback: try to access as tuple/list
                        intersection_points.append((float(point[0]), float(point[1])))
                elif hasattr(intersection, 'x') and hasattr(intersection, 'y'):
                    # Intersection object directly has x, y attributes
                    intersection_points.append((float(intersection.x), float(intersection.y)))
                else:
                    # Fallback: assume intersection is already a coordinate
                    intersection_points.append((float(intersection[0]), float(intersection[1])))
            except (AttributeError, TypeError, IndexError) as e:
                # Skip malformed intersection objects
                print(f"Warning: Could not extract coordinates from intersection object: {e}")
                continue

        return intersection_points

    def add_path(self, confirmed_path: List[Tuple[int, int]], layer: int):
        """
        Add a successfully routed path to the crossing detector.

        Args:
            confirmed_path: List of (x, y) coordinates forming the path
            layer: Layer ID to add the path to
        """
        if len(confirmed_path) < 2:
            return

        # Create segments from the path and add to layer
        for i in range(len(confirmed_path) - 1):
            segment = (confirmed_path[i], confirmed_path[i + 1])
            self.layer_segments[layer].append(segment)

    def clear_layer(self, layer: int):
        """Clear all paths on a specific layer."""
        if layer in self.layer_segments:
            del self.layer_segments[layer]

    def clear_all(self):
        """Clear all paths from all layers."""
        self.layer_segments.clear()

    def get_layer_statistics(self, layer: int) -> Dict[str, int]:
        """Get statistics for a specific layer."""
        return {
            'segments': len(self.layer_segments[layer])
        }