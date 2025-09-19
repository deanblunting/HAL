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
        if len(proposed_path) < 2:
            return False

        # Get existing segments on this layer
        existing_segments = self.layer_segments[layer]
        if not existing_segments:
            return False  # No existing paths to cross

        # Create segments from proposed path
        proposed_segments = []
        for i in range(len(proposed_path) - 1):
            proposed_segments.append((proposed_path[i], proposed_path[i + 1]))

        # Filter out degenerate segments (same start/end point)
        valid_existing = [seg for seg in existing_segments if seg[0] != seg[1]]
        valid_proposed = [seg for seg in proposed_segments if seg[0] != seg[1]]

        if not valid_existing or not valid_proposed:
            return False

        # Combine all segments and use library to check for intersections
        all_segments = valid_existing + valid_proposed

        # Use the library's Bentley-Ottmann algorithm
        intersections = lsi.bentley_ottman(all_segments)
        # If there are any intersections, we have a crossing
        # Note: The library only reports actual intersections, not endpoint sharing
        return len(intersections) > 0

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
        # Skip crossing detection if points are the same (no segment)
        if p1 == p2:
            return False

        # Get existing segments on this layer
        existing_segments = self.layer_segments[layer]
        if not existing_segments:
            return False

        # Skip degenerate segments
        if p1 == p2:
            return False

        # Filter out degenerate segments from existing
        valid_existing = [seg for seg in existing_segments if seg[0] != seg[1]]

        if not valid_existing:
            return False

        # Create test segment and combine with existing
        test_segment = (p1, p2)
        all_segments = valid_existing + [test_segment]

        # Use the library to check for intersections
        intersections = lsi.bentley_ottman(all_segments)
        # If there are intersections, this segment would cross existing ones
        return len(intersections) > 0

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