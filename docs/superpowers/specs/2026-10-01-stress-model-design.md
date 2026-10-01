# Stress model, and models bound to a graph

## Goal

Add stress (Kamada–Kawai) as a third choice in the app's preset menu, next to Balloon and Dense.

The power law models (`docs/superpowers/specs/2026-10-01-repulsion-models-design.md`) each fail on part of the graphs:

- `1/d` (Balloon) makes trees ugly.
- `1/d²` (Dense) converges poorly.

Stress puts a spring between every pair of nodes, with a rest length of the pair's hop count × `EDGE_LENGTH`. It replaces both the edge springs and the repulsion, and is expected to lay out trees and wireframe bodies alike, and to converge reliably.

To make room for it, a model is split from its *field*: a model is a set of knobs, and a field is a model bound to one graph. The field holds anything computed once per graph (the hop table for stress) and computes all forces and the energy.

## Out of scope

- The fixes from the power law review:
  - `expm1` near exponent 1
  - arrow caps on the power law spinboxes
  - the spec sentence on overlap energy for p < 1
  - `model=BALLOON` in the tension test
  - a docstring for `with_model`

  These are a separate commit.
- How `improved` handles a current energy of +∞. Stress cannot reach it; power law layouts still can, as before.
- Saving the chosen model or knob values between runs.
- Changes to `improved`, jitter, pinning or the drawing.

## Models and fields

### Interface

A **model** is a frozen dataclass of knobs.

- It checks its constraints in `__post_init__` and raises `ValueError` with a message naming the knob. The checks are written so that `nan` breaks them.
- Each knob has a comment giving its meaning and constraint.
- Models are compared by value, which the preset menu uses.
- `model.bound_to(edges)` returns a field for the graph given by `edges`, the same per-node neighbour lists `GraphLayout` takes.

A **field** is a model bound to one graph.

- `field.model` is the model it was bound from.
- `field.delta_and_energy(locations)` returns `(forces, energy)`:
  - `forces` is the `(n, 2)` array of the negative gradient of `energy`;
  - `energy` is a float.
- The field does not know about pins.

### `PowerLaw` and `PowerLawField`

`PowerLaw` keeps its knobs, constraints and the presets `BALLOON` and `DENSE` unchanged. It gains `bound_to(edges)`, returning a `PowerLawField`.

`PowerLawField.delta_and_energy` holds the code that is in `GraphLayout.calculate_delta_and_energy` now, unchanged in effect:

- the per-node loop
- edge springs
- the model's repulsion and energy

`GraphLayout.attraction` becomes a module-level function `attraction(loc_deltas)`, used by `PowerLawField`.

`PowerLaw.repulsion` and `PowerLaw.energy` stay as they are.

### `Stress` and `StressField`

```python
@dataclasses.dataclass(frozen=True)
class Stress:
    # how much distant pairs count: a pair h hops apart has the weight h**-weight_exponent;
    # 0 makes every pair count the same, 2 (the usual Kamada–Kawai choice) lets
    # the nearby structure dominate;
    # 0 or more
    weight_exponent: float
```

Constraint: finite and `>= 0`. Message: `weight exponent: a number, 0 or more`.

Preset: `STRESS = Stress(weight_exponent=2)`, with a comment saying that every pair is held at its hop count × `EDGE_LENGTH`, so trees and meshes keep their shape.

`Stress.bound_to(edges)` returns a `StressField` holding:

- `hops`: an `(n, n)` array of hop counts, from a breadth-first search from every node. Duplicate edges and self-loops do not change it.
  - The diagonal is 0 and is never used.
  - Pairs in different components get `max hop count + 1`, where the maximum is over pairs in the same component.
  - A graph without edges, where every pair is in a different component, gets 1 everywhere off the diagonal.
- `rest_lengths = hops * EDGE_LENGTH` and `weights = hops ** -weight_exponent`, both computed once when binding, off the diagonal.

`StressField.delta_and_energy(locations)` works on all pairs at once with numpy, without a loop over nodes.

- Energy: `Σ over pairs i < j of weights * (d - rest_lengths)**2 / (4 * EDGE_LENGTH)`. For two neighbours this is the current spring term.
- Force on node i: `Σ over j of weights * (d - rest_lengths) * (x_j - x_i) / (2 * EDGE_LENGTH * d)`. That's the current attraction formula, per pair.
- Two nodes on the same spot (`d = 0`): their pair adds its finite energy `weights * rest_lengths**2 / (4 * EDGE_LENGTH)` and no force (0/0 is dropped, as `nansum` does now). The result has no `nan` and no `inf`.

Memory: for Pipe2000 (2000 nodes) each `(n, n)` float array is 32 MB. The field keeps three of them, and one evaluation creates a few temporary ones.

### `GraphLayout`

```python
GraphLayout(edges, locations, pinned=None, model=BALLOON, field=None)
```

- When `field` is `None`, it binds `model`: `field = model.bound_to(edges)`. When `field` is given, `model` is ignored.
- `self.field` holds the field. `self.model` is a property returning `self.field.model`.
- `delta` and `energy` come from `self.field.delta_and_energy(self.locations)`. Then the forces on pinned nodes are set to 0, as now, and the tension is computed as now.
- `moved()` passes `self.field`, so `step`, `pinned_at`, `unpinned`, `unpinned_all`, `randomized_layout` and `jittered` reuse the field and its hop table.
- `with_model(model)` binds the new model to the same edges.

## UI

### Presets

`PRESETS` in `app.py`:

| menu entry | model |
|---|---|
| Balloon (1/d) | `BALLOON` |
| Dense (1/d²) | `DENSE` |
| Stress (hops) | `STRESS` |

"Custom" is shown, as now, when the model matches no preset.

### Knob rows follow the model type

A table in `app.py` lists the knobs of each model type:

```python
# model type: [(field, label, lowest arrow value, increment, highest arrow value)];
# typed values beyond the arrows' range are allowed, the model checks them
KNOBS = {
    PowerLaw: [
        ('strength', "Strength", 0, 0.1, 1000),
        ('exponent', "Exponent", 0.1, 0.1, 1000),
    ],
    Stress: [
        ('weight_exponent', "Weight exponent", 0, 0.1, 10),
    ],
}
```

The power law rows keep their current values; changing them belongs to the separate fix commit.

The group:

- **Title:** "Forces" instead of "Repulsion", since stress replaces the springs as well.
- **Rows:** the preset menu stays in the first row. Below it are the knob rows of the current model's type, then the red message row.
- **Changing type:** when the model's type changes (a preset of another type is picked), the knob rows are destroyed and rebuilt from `KNOBS`, and the knob messages are cleared.
- **Same type:** within a type, the spinbox values are set as now. The order stays: the model is applied first, then the spinboxes are set, so their traces find the model unchanged.

`model_with_knob` stays as it is; it works for any model through `dataclasses.replace`.

### Switching while running

Picking a model or changing a knob calls `self.layout.with_model(model)`. That rebinds the field and builds the hop table for stress. Positions, the iteration count and the jitter carry on, and the status-line energy jumps.

## Tests

`tests/test_layout.py`:

- **Hop table:**
  - a path of 4: `hops[0, 3] == 3`
  - a cycle of 6: the opposite nodes are 3 apart
  - a star: leaves are 2 apart
  - two separate edges: pairs across them get `1 + 1 = 2`
  - a graph of 3 nodes without edges: 1 everywhere off the diagonal
  - duplicate edges and a self-loop do not change the table
- **Stress energy:**
  - two neighbours at distance d have `(d - EDGE_LENGTH)**2 / (4 * EDGE_LENGTH)`, the current spring term
  - on a path of 3, nodes 0 and 2 add `(1/4) * (d - 2 * EDGE_LENGTH)**2 / (4 * EDGE_LENGTH)` at weight exponent 2
- **Gradient:** forces are the negative gradient of the energy, by finite differences, for weight exponents 0, 1 and 2, on a path and on a graph of two components.
- **Overlap:** two nodes on the same spot give a finite energy, and `delta` has no `nan`.
- **Constraints:** `Stress` rejects `-0.1`, `nan` and `inf`, with the knob named in the message, and accepts 0.
- **Binding:**
  - every derived layout (`step`, `pinned_at`, `unpinned`, `unpinned_all`, `randomized_layout`, `jittered`, both `toggle_pin` branches) has the same `field` object as its source
  - `with_model` gives a new field with the new model
- **Power law:** the existing tests stay and pass unchanged, except that `layout.attraction(...)` becomes `attraction(...)`.

`tests/test_app.py`:

- `preset_name(STRESS)` returns "Stress (hops)".
- For each model type in `KNOBS`, the listed field names equal the dataclass's field names, in order.

Not checked without a display: the widgets, including rebuilding the knob rows when the model type changes. The user checks these in the app.
