# Multi-Object Tracking: A Learning Guide

This guide explains the concepts behind multi-object tracking for LiDAR point clouds. The goal is to help you understand **WHY** things work, not just what to implement.

---

## Table of Contents
1. [The Tracking Problem](#the-tracking-problem)
2. [Phase 1: Detections](#phase-1-detections)
3. [Phase 2: IoU - Measuring Overlap](#phase-2-iou---measuring-overlap)
4. [Phase 3: The Assignment Problem](#phase-3-the-assignment-problem)
5. [Phase 4: Track Lifecycle](#phase-4-track-lifecycle)
6. [Phase 5: Kalman Filter](#phase-5-kalman-filter)
7. [Putting It Together](#putting-it-together)
8. [Testing Your Implementation](#testing-your-implementation)
9. [Common Pitfalls](#common-pitfalls)

---

## The Tracking Problem

### Detection vs Tracking

**Detection** answers: "Where are objects in THIS frame?"
- Input: Point cloud
- Output: List of clusters with positions

**Tracking** answers: "Which object in frame N is the SAME as which object in frame N-1?"
- Input: Detections from multiple frames
- Output: Persistent identities across time

### Why This Is Hard

Consider two cars side by side. In frame 1, your clustering finds them. In frame 2, it finds them again. But the cluster IDs are arbitrary - DBSCAN doesn't know that "cluster 0 in frame 1" should match "cluster 2 in frame 2".

Tracking solves this by assigning **track IDs** that persist across frames.

### The Core Insight

Objects obey physics. They can't teleport. Between consecutive frames (typically 100ms apart), a car at highway speed moves only ~3 meters. This means:

1. An object's position in frame N should be CLOSE to its position in frame N-1
2. We can use spatial proximity to match detections to tracks
3. The closer two boxes are, the more likely they're the same object

---

## Phase 1: Detections

### What You're Building

A `Detection` is a structured representation of one cluster from one frame. Currently, after DBSCAN, you have:
- `labels`: An array where `labels[i]` tells you which cluster point `i` belongs to
- `obstacle_points`: The actual XYZ coordinates

This is inconvenient for tracking. You want per-cluster summaries.

### The Bounding Box

For each cluster, compute an axis-aligned bounding box:

```
Given points belonging to cluster k:
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, z_max = points.max(axis=0)
```

**Why axis-aligned?** Simpler math. Oriented boxes (rotated to fit the object) are more accurate but require rotation handling. Start simple.

### The Centroid

The center of mass of the cluster:

```
centroid = points.mean(axis=0)  # Simple average of all points
```

**Why centroid matters:** It's a single point representing "where the object is". Useful for distance calculations and Kalman state.

### Data Structure Hints

Your `Detection` should store:
- Which frame it came from
- The bounding box (min/max corners, or derived center + dimensions)
- The centroid
- Number of points (useful for confidence estimation)
- Optionally: the actual points (memory-heavy, usually not needed)

Your `BoundingBox` should support:
- Computing area (for 2D BEV) or volume (for 3D)
- Computing center
- Extracting min/max per axis

### Exercise: Extract Detections

Write a function that takes `obstacle_points` and `cluster_labels`, and returns a list of `Detection` objects. Loop through unique labels (skip -1 for noise), extract points for each, compute bbox and centroid.

---

## Phase 2: IoU - Measuring Overlap

### The Intuition

IoU (Intersection over Union) answers: "How much do these two boxes overlap?"

- IoU = 1.0: Boxes are identical
- IoU = 0.0: Boxes don't touch at all
- IoU = 0.5: Significant overlap, probably same object

### Why IoU Works for Tracking

If an object barely moves between frames, its bounding box in frame N will heavily overlap with frame N-1. High IoU = likely same object.

### 2D BEV IoU (Bird's Eye View)

Start with 2D - ignore the Z axis. For vehicles on a road, height doesn't change much. This simplifies the math to rectangle intersection.

**The Algorithm:**

Given two rectangles, each defined by (x_min, y_min, x_max, y_max):

```
Step 1: Find intersection rectangle
    inter_x_min = max(box1.x_min, box2.x_min)
    inter_y_min = max(box1.y_min, box2.y_min)
    inter_x_max = min(box1.x_max, box2.x_max)
    inter_y_max = min(box1.y_max, box2.y_max)

Step 2: Compute intersection area (0 if no overlap)
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection = inter_width * inter_height

Step 3: Compute union area
    area1 = (box1.x_max - box1.x_min) * (box1.y_max - box1.y_min)
    area2 = (box2.x_max - box2.x_min) * (box2.y_max - box2.y_min)
    union = area1 + area2 - intersection

Step 4: IoU
    iou = intersection / union  (handle union=0 edge case)
```

### Visual Understanding

```
    +-------+
    |   A   |
    |   +---+---+
    |   |###|   |   ### = intersection
    +---+---+   |
        |   B   |
        +-------+

IoU = (area of ###) / (area of A + area of B - area of ###)
```

### Why max(0, ...) Matters

If boxes don't overlap, `inter_x_max < inter_x_min`. Without `max(0, ...)`, you'd get negative width, leading to negative intersection. The max clamps this to 0.

### Edge Cases

- **Identical boxes:** intersection = area, union = area, IoU = 1.0
- **No overlap:** intersection = 0, IoU = 0.0
- **One inside other:** intersection = smaller area, union = larger area
- **Zero-area box:** Handle explicitly to avoid division by zero

### Exercise: Implement IoU

Write `compute_iou(box1, box2) -> float`. Test with known cases:
- Two identical unit squares → 1.0
- Two squares 10 units apart → 0.0
- Two squares overlapping by half in X → calculate expected value by hand, verify

---

## Phase 3: The Assignment Problem

### The Setup

You have:
- N detections in the current frame
- M existing tracks from previous frames

Goal: Find the best matching between detections and tracks.

### Why Greedy Fails

**Greedy approach:** For each detection, assign it to the track with highest IoU.

**Problem:**
```
Detection A: IoU with Track 1 = 0.6, Track 2 = 0.5
Detection B: IoU with Track 1 = 0.55, Track 2 = 0.1

Greedy: A→1 (0.6), B→2 (0.1), total IoU = 0.7
Optimal: A→2 (0.5), B→1 (0.55), total IoU = 1.05
```

Greedy "steals" Track 1 for A, leaving B with a bad match. The optimal solution gives both a decent match.

### The Cost Matrix

Build a matrix where `cost[i][j]` = cost of assigning detection i to track j.

Since we want HIGH IoU to mean LOW cost (Hungarian minimizes):
```
cost[i][j] = 1.0 - iou(detection_i, track_j)
```

Example with 2 detections and 3 tracks:
```
           Track0  Track1  Track2
Detection0 [ 0.3    0.8     0.9  ]   (IoUs were 0.7, 0.2, 0.1)
Detection1 [ 0.4    0.5     0.2  ]   (IoUs were 0.6, 0.5, 0.8)
```

### Hungarian Algorithm

Also called the "Kuhn-Munkres algorithm" or "linear assignment problem".

**What it does:** Finds the assignment that minimizes total cost, with the constraint that each detection matches at most one track, and vice versa.

**You don't need to implement it.** Use scipy:
```python
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost_matrix)
# row_ind[i] is matched to col_ind[i]
```

### Applying a Threshold

Not all assignments are valid. If best IoU for a detection is 0.05, that's probably not the same object.

After Hungarian matching:
```
for each (det_idx, track_idx) pair:
    if cost[det_idx][track_idx] > (1 - iou_threshold):
        # IoU too low, reject this match
        mark detection as unmatched
        mark track as unmatched
```

Typical threshold: 0.3 (so minimum IoU of 0.3 to accept)

### Handling Unequal Counts

What if you have 5 detections but only 3 tracks? Or 2 detections and 4 tracks?

**More detections than tracks:** Some detections are unmatched → create new tracks
**More tracks than detections:** Some tracks are unmatched → objects left scene or missed detection

Hungarian handles rectangular matrices. It just leaves some items unassigned.

### Exercise: Build the Matching Logic

1. Write a function that builds the cost matrix from detections and tracks
2. Call `linear_sum_assignment`
3. Filter matches by threshold
4. Return three lists: matched pairs, unmatched detection indices, unmatched track indices

---

## Phase 4: Track Lifecycle

### The Track Object

A `Track` represents a persistent object identity. It needs:

**Identity:**
- `track_id`: Globally unique, never reused (use a counter)

**State:**
- Current position estimate (centroid or full Kalman state)
- Current bounding box estimate

**History:**
- `age`: How many frames since creation
- `hits`: How many frames with successful matches
- `misses`: Consecutive frames without a match (resets on match)

**Status:**
- `TENTATIVE`: Just created, might be noise
- `CONFIRMED`: Reliable, has enough hits
- `LOST`: Too many consecutive misses, ready for deletion

### Track State Transitions

```
                    ┌─────────────┐
      new detection │  TENTATIVE  │
         creates    └──────┬──────┘
                           │
            hits >= n_init │
                           ▼
                    ┌─────────────┐
                    │  CONFIRMED  │◄──────┐
                    └──────┬──────┘       │ matched
                           │              │ (reset misses)
         misses > max_age  │              │
                           ▼              │
                    ┌─────────────┐       │
                    │    LOST     │───────┘
                    └─────────────┘   (if matched before lost)
                           │
                           ▼
                        DELETE
```

### Why Track States Matter

**TENTATIVE:** A single detection could be noise (false positive from clustering). Wait for `n_init` (e.g., 3) consecutive matches before trusting it.

**CONFIRMED:** This is a real object. Display it to the user.

**LOST:** The object might have left the scene, or detection temporarily failed. Keep the track for `max_age` (e.g., 5) frames in case it reappears.

### The Update Cycle

Each frame:

```
1. For each track: predict new position (trivial without Kalman: just keep last position)

2. Match detections to tracks (Phase 3)

3. For matched pairs:
   - Update track position with detection
   - Increment hits
   - Reset misses to 0
   - If TENTATIVE and hits >= n_init: transition to CONFIRMED

4. For unmatched tracks:
   - Increment misses
   - If misses > max_age: transition to LOST

5. For unmatched detections:
   - Create new track (status = TENTATIVE)
   - Assign new unique track_id

6. Remove all LOST tracks from active list

7. Return CONFIRMED tracks for visualization
```

### Choosing Parameters

- `iou_threshold = 0.3`: Minimum IoU to accept match. Lower = more lenient, risks merging different objects. Higher = stricter, risks breaking tracks.

- `n_init = 3`: Frames to confirm. Higher = fewer false tracks, but slow to confirm new objects.

- `max_age = 5`: Frames to keep lost track. Higher = better recovery from occlusion, but ghosts persist longer.

Start with these values, tune based on your data.

### Exercise: Implement Track Management

1. Create a `Track` class with the fields described
2. Create a `Tracker` class that maintains a list of tracks and a `next_id` counter
3. Implement the `update(detections)` method following the cycle above
4. Test with synthetic data: create detections that move slightly each frame, verify track IDs are stable

---

## Phase 5: Kalman Filter

### Why Prediction Helps

Without prediction, you compare current detection position to LAST KNOWN track position. If an object moves fast, boxes may not overlap at all.

With prediction, you estimate WHERE THE OBJECT SHOULD BE NOW based on its velocity. Then compare detection to PREDICTED position.

```
Without Kalman:
  Frame N-1: Object at (10, 0)
  Frame N:   Object at (15, 0)  ← moved 5 units
  Track position: (10, 0)       ← still at old position
  IoU: Boxes might not overlap!

With Kalman (velocity = 5 units/frame):
  Frame N-1: Object at (10, 0), velocity (5, 0)
  Frame N:   Object at (15, 0)
  Predicted position: (15, 0)   ← predicts the movement!
  IoU: High overlap, match succeeds
```

### State Vector

Minimum state for 3D tracking:
```
state = [x, y, z, vx, vy, vz]
         └─position─┘ └─velocity─┘
```

**x, y, z:** Current position estimate
**vx, vy, vz:** Current velocity estimate

### The Prediction Step

Assuming constant velocity (simplest motion model):

```
new_x = old_x + vx * dt
new_y = old_y + vy * dt
new_z = old_z + vz * dt
```

In matrix form: `x_new = F @ x_old`

Where F is the state transition matrix:
```
F = [1, 0, 0, dt, 0,  0 ]
    [0, 1, 0, 0,  dt, 0 ]
    [0, 0, 1, 0,  0,  dt]
    [0, 0, 0, 1,  0,  0 ]
    [0, 0, 0, 0,  1,  0 ]
    [0, 0, 0, 0,  0,  1 ]
```

For frame-by-frame tracking, dt = 1 (one frame).

### The Update Step

When you match a detection to a track, you have a MEASUREMENT (the detection's centroid). Use it to CORRECT the prediction.

The Kalman gain (K) determines how much you trust the measurement vs the prediction:
- High K: Trust measurement more (responsive but noisy)
- Low K: Trust prediction more (smooth but slow to react)

The filter automatically computes K based on:
- **Process noise (Q):** How much the object might deviate from constant velocity
- **Measurement noise (R):** How uncertain your detection centroid is

### Simplified Kalman Equations

```
PREDICT:
  x_pred = F @ x                    # Predict state
  P_pred = F @ P @ F.T + Q          # Predict uncertainty

UPDATE (when matched):
  y = z - H @ x_pred                # Innovation (measurement - prediction)
  S = H @ P_pred @ H.T + R          # Innovation covariance
  K = P_pred @ H.T @ inv(S)         # Kalman gain
  x = x_pred + K @ y                # Updated state
  P = (I - K @ H) @ P_pred          # Updated uncertainty
```

Where:
- `x`: State vector [x, y, z, vx, vy, vz]
- `P`: State covariance (6x6 matrix representing uncertainty)
- `F`: State transition matrix (prediction model)
- `H`: Measurement matrix (extracts [x, y, z] from state)
- `Q`: Process noise covariance (how much model deviates from reality)
- `R`: Measurement noise covariance (how noisy detections are)
- `z`: Measurement (detected centroid [x, y, z])

### Handling Missed Detections

When a track is NOT matched:
- Still run PREDICT (track coasts forward)
- Skip UPDATE (no measurement to correct with)
- Uncertainty (P) grows each frame without update

This lets tracks "coast" through brief occlusions.

### Practical Starting Values

```
# Process noise - assumes up to 1 m/s^2 acceleration
Q = diag([0.1, 0.1, 0.1, 1.0, 1.0, 1.0])  # position, velocity

# Measurement noise - centroid uncertainty ~0.5m
R = diag([0.25, 0.25, 0.25])  # (0.5)^2 for each axis

# Initial velocity uncertainty (don't know initial velocity)
Initial P velocity terms: 100 (high uncertainty)
```

### When to Use Kalman

- **Slow objects, high frame rate:** IoU alone works fine. Kalman optional.
- **Fast objects, low frame rate:** Kalman essential. Boxes won't overlap without prediction.
- **Occlusions:** Kalman lets tracks coast through gaps.

### Exercise: Add Kalman

1. Extend Track to store Kalman state (x, P)
2. Implement `predict()` method
3. Implement `update(detection)` method
4. Modify IoU computation to use PREDICTED bbox, not last known
5. Test: Create detections moving at constant velocity. Verify tracker handles gaps (missing frames).

---

## Putting It Together

### The Tracker Class

```
class Tracker:
    tracks: List[Track]
    next_id: int
    config: (iou_threshold, max_age, n_init, use_kalman)

    def update(self, detections: List[Detection]) -> List[Track]:
        # 1. Predict
        for track in self.tracks:
            track.predict()

        # 2. Associate
        matches, unmatched_dets, unmatched_tracks = associate(
            detections, self.tracks, self.config.iou_threshold
        )

        # 3. Update matched tracks
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 4. Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # 5. Create new tracks
        for det_idx in unmatched_dets:
            new_track = Track(
                track_id=self.next_id,
                detection=detections[det_idx],
                n_init=self.config.n_init,
                max_age=self.config.max_age
            )
            self.next_id += 1
            self.tracks.append(new_track)

        # 6. Prune lost tracks
        self.tracks = [t for t in self.tracks if t.status != LOST]

        # 7. Return confirmed
        return [t for t in self.tracks if t.status == CONFIRMED]
```

### Integration with Pipeline

```
# Existing pipeline (per frame)
points = load_kitti_txt(frame_path)
downsampled = voxel_downsample(points)
filtered = radial_outlier_removal(downsampled)
_, ground_mask = ransac_ground_plane(filtered)
obstacle_points = filtered[~ground_mask]
cluster_result = dbscan_cluster(obstacle_points)

# NEW: Convert to detections
detections = create_detections(obstacle_points, cluster_result, frame_idx)

# NEW: Update tracker
active_tracks = tracker.update(detections)

# active_tracks now has persistent IDs across frames
```

---

## Testing Your Implementation

### Unit Tests for IoU

```python
def test_iou_identical():
    box = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    assert compute_iou(box, box) == 1.0

def test_iou_no_overlap():
    box1 = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    box2 = BoundingBox(x_min=5, y_min=5, x_max=6, y_max=6)
    assert compute_iou(box1, box2) == 0.0

def test_iou_partial():
    # Two unit squares, overlapping by 0.5 in X
    box1 = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    box2 = BoundingBox(x_min=0.5, y_min=0, x_max=1.5, y_max=1)
    # Intersection: 0.5 * 1 = 0.5
    # Union: 1 + 1 - 0.5 = 1.5
    # IoU: 0.5 / 1.5 = 0.333...
    assert abs(compute_iou(box1, box2) - 1/3) < 0.001
```

### Unit Tests for Tracker

```python
def test_single_detection_creates_track():
    tracker = Tracker()
    det = Detection(centroid=[0, 0, 0], bbox=...)
    tracks = tracker.update([det])
    # First frame: track is TENTATIVE, not returned
    assert len(tracks) == 0
    assert len(tracker.tracks) == 1

def test_track_confirmed_after_n_init():
    tracker = Tracker(n_init=3)
    det = Detection(centroid=[0, 0, 0], bbox=...)

    for _ in range(3):
        tracker.update([det])

    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracks[0].status == CONFIRMED

def test_stationary_object_same_id():
    tracker = Tracker(n_init=1)
    det = Detection(centroid=[0, 0, 0], bbox=...)

    tracker.update([det])
    tracks = tracker.update([det])

    assert tracks[0].track_id == 0  # Same ID both frames
```

### Integration Test

Write a script that:
1. Loads 5-10 frames from your KITTI sequence
2. Runs the full pipeline on each
3. Prints track IDs and centroids per frame
4. Verify visually that stationary objects keep consistent IDs

---

## Common Pitfalls

### 1. Confusing cluster_id and track_id

**Symptom:** Track IDs change every frame
**Cause:** Using cluster_id (from DBSCAN) instead of track_id (from tracker)
**Fix:** Detections have cluster_id (per-frame), Tracks have track_id (persistent)

### 2. Mutating Shared References

**Symptom:** Track history is corrupted, all entries show same values
**Cause:** Storing reference to detection, then modifying it
**Fix:** Deep copy detections when storing in track history

```python
# Bad
self.history.append(detection)

# Good
import copy
self.history.append(copy.deepcopy(detection))
```

### 3. Forgetting Edge Cases

**Symptom:** Crashes or weird behavior
**Cause:** Not handling empty detection lists, zero-area boxes, etc.
**Fix:** Add explicit checks:

```python
if len(detections) == 0:
    # Mark all tracks as missed, return empty
    ...

if box.area() == 0:
    return 0.0  # No overlap possible
```

### 4. IoU Threshold Too Strict

**Symptom:** Track IDs increment rapidly, even for stationary objects
**Cause:** Threshold too high, boxes with minor differences rejected
**Fix:** Lower threshold (try 0.2 instead of 0.5)

### 5. Not Using Predicted Position

**Symptom:** Tracking works for stationary objects but fails for moving ones
**Cause:** Computing IoU with last known position instead of predicted
**Fix:** Always call `track.predict()` before building cost matrix, use predicted bbox

### 6. Track Never Confirms

**Symptom:** Tracks exist but none are CONFIRMED
**Cause:** `n_init` too high, or detections not consistent enough
**Fix:** Lower `n_init`, or debug why detections don't match (print IoU values)

---

## Next Steps After Basic Tracking

Once you have IoU + Kalman working:

1. **Oriented Bounding Boxes:** Fit rotated boxes to clusters for better IoU on angled vehicles

2. **Multi-hypothesis Tracking:** Handle ambiguous situations by maintaining multiple possible assignments

3. **Re-identification:** Match tracks that were lost and reappear (use appearance features or motion patterns)

4. **Classification:** Different tracking parameters for cars vs pedestrians vs cyclists

5. **Trajectory Smoothing:** Post-process tracks to remove jitter

---

Good luck! The key to learning is implementing each piece yourself, testing it thoroughly, and understanding WHY it works before moving to the next phase.
