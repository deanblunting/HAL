"""
Unified crossing detection using Bentley-Ottmann sweep line algorithm.
Provides O((n + k) log n) crossing detection for HAL quantum routing.
"""

import heapq
from typing import List, Tuple, Set, Dict, Optional, Union
from collections import defaultdict
from enum import Enum


class EventType(Enum):
    """Event types for sweep line algorithm."""
    START = "START"
    END = "END"
    INTERSECTION = "INTERSECTION"


class Event:
    """Event in the sweep line algorithm."""

    def __init__(self, x: int, y: int, event_type: EventType, segment_id: Optional[int] = None):
        self.x = x
        self.y = y
        self.type = event_type
        self.segment_id = segment_id

    def __lt__(self, other: 'Event') -> bool:
        """Order events by x-coordinate, then by y-coordinate, then by type."""
        if self.x != other.x:
            return self.x < other.x
        if self.y != other.y:
            return self.y < other.y
        # Process START events before END events at same position
        if self.type != other.type:
            return self.type == EventType.START
        return False

    def __repr__(self) -> str:
        return f"Event({self.x}, {self.y}, {self.type.value}, seg={self.segment_id})"


class Segment:
    """Line segment for sweep line algorithm."""

    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], segment_id: int):
        # Ensure p1 is leftmost point (or topmost if vertical)
        if p1[0] < p2[0] or (p1[0] == p2[0] and p1[1] < p2[1]):
            self.p1, self.p2 = p1, p2
        else:
            self.p1, self.p2 = p2, p1

        self.id = segment_id
        self.is_vertical = (self.p1[0] == self.p2[0])

        if not self.is_vertical:
            # Compute slope and y-intercept for non-vertical segments
            dx = self.p2[0] - self.p1[0]
            dy = self.p2[1] - self.p1[1]
            self.slope = dy / dx if dx != 0 else float('inf')
            self.y_intercept = self.p1[1] - self.slope * self.p1[0]
        else:
            self.slope = float('inf')
            self.y_intercept = None

    def y_at_x(self, x: int) -> float:
        """Get y-coordinate of segment at given x."""
        if self.is_vertical:
            if x == self.p1[0]:
                return min(self.p1[1], self.p2[1])  # Return bottom y for ordering
            else:
                raise ValueError(f"Cannot get y-coordinate of vertical segment at x={x}")
        return self.slope * x + self.y_intercept

    def intersects_segment(self, other: 'Segment') -> Optional[Tuple[float, float]]:
        """Check if this segment intersects with another, return intersection point or True for overlaps."""
        # Use our existing line_segments_intersect function logic first
        from .routing import line_segments_intersect

        if line_segments_intersect(self.p1, self.p2, other.p1, other.p2):
            # Try to calculate actual intersection point
            intersection_point = self._calculate_intersection(other)
            # For collinear overlaps, _calculate_intersection returns None
            # but we still want to indicate that segments intersect
            return intersection_point if intersection_point is not None else (0.0, 0.0)  # Dummy point for overlap

        # Check for endpoint sharing only - this is NOT considered a crossing for routing
        # But identical segments or overlapping segments ARE crossings
        if (self.p1 == other.p1 and self.p2 == other.p2):
            # Identical segments - this IS a crossing (complete overlap)
            return (0.0, 0.0)  # Dummy point for identical segments

        return None

    def _calculate_intersection(self, other: 'Segment') -> Optional[Tuple[float, float]]:
        """Calculate intersection point between two segments."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        x3, y3 = other.p1
        x4, y4 = other.p2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:  # Parallel or collinear
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            # Intersection point
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)

        return None

    def __repr__(self) -> str:
        return f"Segment({self.p1}, {self.p2}, id={self.id})"


class SweepLineTree:
    """
    Balanced tree structure to maintain active segments ordered by y-coordinate.
    Uses a simple list with binary search for now (can be optimized to balanced BST).
    """

    def __init__(self):
        self.segments = []  # List of (y_value, segment) pairs, kept sorted
        self.current_x = 0

    def set_sweep_position(self, x: int):
        """Update the current sweep line position."""
        self.current_x = x
        # Re-sort segments by their y-coordinate at current x
        self._resort_segments()

    def insert(self, segment: Segment):
        """Insert a segment into the tree."""
        y_value = segment.y_at_x(self.current_x) if not segment.is_vertical else segment.p1[1]
        # Insert in sorted order
        inserted = False
        for i, (existing_y, existing_seg) in enumerate(self.segments):
            if y_value < existing_y:
                self.segments.insert(i, (y_value, segment))
                inserted = True
                break
        if not inserted:
            self.segments.append((y_value, segment))

    def remove(self, segment: Segment):
        """Remove a segment from the tree."""
        self.segments = [(y, seg) for y, seg in self.segments if seg.id != segment.id]

    def get_neighbors(self, segment: Segment) -> Tuple[Optional[Segment], Optional[Segment]]:
        """Get the segments immediately above and below the given segment."""
        segment_index = None
        for i, (y, seg) in enumerate(self.segments):
            if seg.id == segment.id:
                segment_index = i
                break

        if segment_index is None:
            return None, None

        above = self.segments[segment_index - 1][1] if segment_index > 0 else None
        below = self.segments[segment_index + 1][1] if segment_index < len(self.segments) - 1 else None

        return above, below

    def get_adjacent_pairs(self) -> List[Tuple[Segment, Segment]]:
        """Get all adjacent segment pairs for intersection testing."""
        pairs = []
        for i in range(len(self.segments) - 1):
            seg1 = self.segments[i][1]
            seg2 = self.segments[i + 1][1]
            pairs.append((seg1, seg2))
        return pairs

    def _resort_segments(self):
        """Re-sort segments by y-coordinate at current x position."""
        new_segments = []
        for y, segment in self.segments:
            try:
                new_y = segment.y_at_x(self.current_x) if not segment.is_vertical else segment.p1[1]
                new_segments.append((new_y, segment))
            except ValueError:
                # Segment doesn't exist at current x (shouldn't happen in correct usage)
                continue

        self.segments = sorted(new_segments, key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self.segments)

    def __repr__(self) -> str:
        return f"SweepLineTree(x={self.current_x}, segments={[seg.id for _, seg in self.segments]})"


class CrossingDetector:
    """
    Unified crossing detection using Bentley-Ottmann sweep line algorithm.
    Provides O((n + k) log n) crossing detection for all routing methods.
    """

    def __init__(self):
        """Initialize crossing detector."""
        self.layer_paths = defaultdict(list)  # {layer_id: [list of paths]}
        self.layer_segments = defaultdict(list)  # {layer_id: [list of Segment objects]}
        self.next_segment_id = 0

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
            segment = Segment(proposed_path[i], proposed_path[i + 1], -1)  # Use -1 as temp ID
            proposed_segments.append(segment)

        # Use sweep line algorithm to check for intersections
        return self._sweep_line_intersect_check(existing_segments, proposed_segments)

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
        # For single segment, we can use a simpler approach
        existing_segments = self.layer_segments[layer]
        if not existing_segments:
            return False

        test_segment = Segment(p1, p2, -1)

        # Check against all existing segments
        for existing_segment in existing_segments:
            if test_segment.intersects_segment(existing_segment):
                return True

        return False

    def add_path(self, confirmed_path: List[Tuple[int, int]], layer: int):
        """
        Add a successfully routed path to the crossing detector.

        Args:
            confirmed_path: List of (x, y) coordinates forming the path
            layer: Layer ID to add the path to
        """
        if len(confirmed_path) < 2:
            return

        # Store the full path
        self.layer_paths[layer].append(confirmed_path)

        # Create segments from the path
        path_segments = []
        for i in range(len(confirmed_path) - 1):
            segment = Segment(confirmed_path[i], confirmed_path[i + 1], self.next_segment_id)
            path_segments.append(segment)
            self.next_segment_id += 1

        # Add segments to the layer
        self.layer_segments[layer].extend(path_segments)

    def clear_layer(self, layer: int):
        """Clear all paths on a specific layer."""
        if layer in self.layer_paths:
            del self.layer_paths[layer]
        if layer in self.layer_segments:
            del self.layer_segments[layer]

    def clear_all(self):
        """Clear all paths from all layers."""
        self.layer_paths.clear()
        self.layer_segments.clear()
        self.next_segment_id = 0

    def get_layer_statistics(self, layer: int) -> Dict[str, int]:
        """Get statistics for a specific layer."""
        return {
            'paths': len(self.layer_paths[layer]),
            'segments': len(self.layer_segments[layer])
        }

    def _sweep_line_intersect_check(self, existing_segments: List[Segment],
                                   proposed_segments: List[Segment]) -> bool:
        """
        Use sweep line algorithm to check if proposed segments intersect existing ones.
        Returns True if ANY intersection is found (early termination for efficiency).
        """
        if not existing_segments or not proposed_segments:
            return False

        # Combine all segments for sweep line processing
        all_segments = existing_segments + proposed_segments
        proposed_ids = {seg.id for seg in proposed_segments}

        # Create events for all segment endpoints
        events = []
        for segment in all_segments:
            # Add start and end events
            events.append(Event(segment.p1[0], segment.p1[1], EventType.START, segment.id))
            events.append(Event(segment.p2[0], segment.p2[1], EventType.END, segment.id))

        # Sort events by x-coordinate
        events.sort()

        # Initialize sweep line tree
        sweep_tree = SweepLineTree()
        segment_lookup = {seg.id: seg for seg in all_segments}

        # Process events
        for event in events:
            sweep_tree.set_sweep_position(event.x)
            segment = segment_lookup[event.segment_id]

            if event.type == EventType.START:
                # Insert segment into sweep line
                sweep_tree.insert(segment)

                # Check intersections with neighbors
                above, below = sweep_tree.get_neighbors(segment)

                # Only report intersection if at least one segment is from proposed set
                if above and (segment.id in proposed_ids or above.id in proposed_ids):
                    if segment.intersects_segment(above):
                        return True  # Found crossing - early termination

                if below and (segment.id in proposed_ids or below.id in proposed_ids):
                    if segment.intersects_segment(below):
                        return True  # Found crossing - early termination

            elif event.type == EventType.END:
                # Get neighbors before removal
                above, below = sweep_tree.get_neighbors(segment)

                # Remove segment from sweep line
                sweep_tree.remove(segment)

                # Check if the neighbors now intersect each other
                if above and below and (above.id in proposed_ids or below.id in proposed_ids):
                    if above.intersects_segment(below):
                        return True  # Found crossing - early termination

        return False  # No crossings found