# Field Spacing Feature Impact Analysis

## Overview
The user has requested the ability to configure distinct spacing between 4x4 panel "Fields" (blocks), separating this from the standard "Grid Pitch" (spacing within a block).

**New Parameters:**
- `field_spacing_x` (default 2.3m): Pivot-to-pivot distance jumping from column 3 of Field `N` to column 0 of Field `N+1`.
- `field_spacing_y` (default 2.4m): Pivot-to-pivot distance jumping from row 3 of Field `N` to row 0 of Field `N+1`.

**Current System:**
- Assumes a uniform infinite grid where `pitch_x` and `pitch_y` are constant everywhere.
- Uses a `3x3 Kernel` to solve physics for the "Center Panel", assuming all 8 neighbors are at fixed relative positions.

---

## Impact on Calculations

### 1. Geometry & Visualization
**Impact: HIGH**
- **Current**: Panel position `(r, c)` is simply `(c * pitch_x, r * pitch_y)`.
- **New**: Panel position must account for "Field Jumps".
    - `field_col = c // 4`
    - `local_col = c % 4`
    - `x = (field_col * gap_stride_x) + (local_col * pitch_x)`
    - Where `gap_stride_x = (3 * pitch_x) + field_spacing_x`
    - *Note: This assumes `field_spacing_x` is the distance between the last pivot of field `i` and first pivot of field `i+1`.*

### 2. Shadowing (Yield)
**Impact: MODERATE to HIGH**
- **Current**: A panel is shaded by its immediate neighbors at dist `pitch`.
- **New**: Panels at the "Edge" of a field will have one neighbor at dist `pitch` (Inner) and one neighbor at dist `field_spacing` (Outer).
- **Effect**:
    - `field_spacing > pitch`: Outer neighbor is farther away.
    - **Result**: **LESS SHADING** (Higher Yield) for edge panels.
    - Ignoring this (using uniform `pitch`) is a **Conservative Estimate** (Safe but under-reports yield).

### 3. Collision Detection (Safety)
**Impact: CRITICAL for accuracy, MODERATE for safety**
- **Current**: Clash check assumes neighbors are at dist `pitch`.
- **New**: Edge panels have more room on one side.
- **Effect**:
    - `field_spacing > pitch`: Edge panels are *less likely* to clash with the "Outer" neighbor.
    - Ignoring this (using uniform `pitch`) is a **Conservative Estimate** (False positives for clashes at field boundaries, but ensures safety).

---

## Proposed Implementation Options

### Option A: Visualization-Only (Fastest, Conservative)
**Description**: Update the 3D Viewport to show the gaps visually, but keep the Physics Engine running on the uniform "Standard Pitch" assumption.
**Pros**:
- Very fast to implement.
- Minimal risk of regressions in physics kernel.
- **Safe**: Yield and Collision estimates are conservative (worst-case scenario).
**Cons**:
- 3D View will show gaps, but shadows might appear "too long" (as if neighbor was closer).
- "False Clashes" reported at boundaries if user pushes tracking limits.

### Option B: "Field-Aware" Kernel (Adaptive)
**Description**: Update `InfiniteKernel` to detect if a panel is an "Edge Panel" and adjust neighbor distances accordingly.
**Details**:
- The 3D Viewport passes the "local index" (0-3) to the kernel.
- Kernel constructs a custom neighborhood (e.g., neighbor to the West is at `field_spacing` instead of `pitch`).
**Pros**:
- Accurate Physics (Yield & Clashes correct).
- "True" representation of the plant.
**Cons**:
- Significant refactor of `InfiniteKernel`.
- Instead of solving 1 "Representative State" per timestep, we must solve **Up to 9** (Corners, Edges, Center) or **16** unique states.
- ~16x Slower Simulation (unless optimized).

### Option C: Periodic 4x4 Simulation (Compromise)
**Description**: Simulate a 4x4 block with Periodic Boundary Conditions (wrapping around with `field_spacing`).
**Pros**: Accurately modeled 16-panel pattern.
**Cons**: Complex implementation, similar cost to Option B.

---

## Recommendation
**Start with Option A (Visualization Only).**
Since the `field_spacing` (2.3m) is larger than `pitch` (1.7m), the uniform assumption is a safe, conservative baseline. It prevents the user from designing a system that relies on the extra gap for safety (which is good practice).
We can then upgrade to **Option B** if "exact yield" becomes critical.
