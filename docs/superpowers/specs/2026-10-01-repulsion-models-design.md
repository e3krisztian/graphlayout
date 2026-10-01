# Repulsion models: power law with presets

## Goal

Let the user change how nodes repel each other while the app runs, so the same graph can be compared under different repulsions.

Experiments on 2026-10-01 showed that no single repulsion suits every graph:

- `1/d` (the current all-pairs `2/d`) gives 3D-looking pictures of wireframe bodies (meshes, cubes, pipes), but trees come out ugly.
- `1/d²` lays out trees well, but converges poorly.

Both are cases of one power law, `c / d^p`. This task adds that power law as a force model with two knobs (strength c, exponent p), plus two named presets for the cases above.

A stress model (Kamada–Kawai) is a planned follow-up. It is not part of this task, but the model interface here is the place it will plug in.

## Out of scope

- Stress, and any model that also replaces the edge springs.
- Changing the springs, `improved`, jitter, or pinning.
- Saving knob settings between runs.

## Force model

### `PowerLaw`

`PowerLaw` is a frozen dataclass in `layout.py`, holding the repulsion knobs:

```python
@dataclasses.dataclass(frozen=True)
class PowerLaw:
    # how hard two nodes push each other apart;
    # 0 or more
    strength: float
    # how fast the push fades with distance: two nodes at distance d push each other
    # apart with strength / d**exponent; a low exponent reaches far and spreads the
    # whole graph, a high one acts mostly between close nodes;
    # above 0
    exponent: float
```

`__post_init__` checks the constraints and raises `ValueError` with a message naming the knob:

- `strength`: finite and `>= 0`. Message: `strength: a number, 0 or more`.
- `exponent`: finite and `> 0`. Message: `exponent: a number above 0`.

The checks are written so that `nan` breaks them, e.g. `not (math.isfinite(x) and x > 0)`.

### Force and energy

For a pair at distance `d`, with `c = strength` and `p = exponent`:

- Force: `c / d^p`, pointing away from the other node.
- Energy: `-c * (d^(1-p) - 1) / (1-p)`, and `-c * ln d` at `p = 1`, which is the limit of the general form.

So the energy has no jump as `p` passes through 1. The force is the negative derivative of the energy, so `delta` stays the negative gradient of the layout's energy and `improved` keeps its energy-based acceptance.

Every pair is seen from both of its nodes, so the per-node sum is halved, as the current code does. Nodes on top of each other have infinite energy for every `p > 0`, as now.

`PowerLaw` has two methods taking the deltas and distances of one node to the other nodes, with the node itself already left out:

- `repulsion(loc_deltas, distances)` returns the `(2,)` force on the node.
- `energy(distances)` returns the node's halved share of the pair energies.

### Presets

```python
# 1/d reaches far: it inflates meshes from the inside, giving wireframe bodies a 3D look
BALLOON = PowerLaw(strength=2, exponent=1)
# 1/d**2 acts close: it spreads trees into clean branches, but converges slowly
DENSE = PowerLaw(strength=1, exponent=2)
```

`BALLOON` is the current behaviour: the force `2/d` and the energy `-2 Σ ln d` are unchanged.

### `GraphLayout`

- `GraphLayout(edges, locations, pinned=None, model=BALLOON)`, stored as `self.model`.
- `calculate_delta_and_energy` keeps computing the springs itself. For the repulsion it calls `self.model.repulsion(...)` and `self.model.energy(...)`.
- New `moved(locations, pinned=None)` creates a layout of the same graph and model at `locations`, keeping the pins unless `pinned` is given. `step`, `pinned_at`, `unpinned`, `unpinned_all`, `randomized_layout` and `jittered` all go through it, so every derived layout keeps the model. `unpinned_all` passes all-`False` pins.
- New `with_model(model)` returns the same layout with another model.
- `GraphLayout.repulsion` moves to `PowerLaw.repulsion`.

## UI

### Repulsion group

A `LabelFrame` "Repulsion" in the toolbar, packed after "Unpin all", with these rows:

1. **Preset**: an option menu with "Balloon (1/d)", "Dense (1/d²)" and "Custom". Picking a preset sets both spinboxes to its values. "Custom" is never applied; the menu shows it when the current model matches no preset.
2. **Strength**: a spinbox from 0, increment 0.1.
3. **Exponent**: a spinbox from 0.1, increment 0.1. Values between 0 and 0.1 can still be typed.
4. **Message**: a red label naming the broken constraints, joined by `; `, or empty.

Both knobs are spinboxes, because neither has a natural upper bound.

### Bad input

Each spinbox's text variable is traced. On every change:

- Text that is empty, not a number, or breaks the constraint turns the spinbox text red, and the message row shows the constraint. The model keeps the last valid value.
- Valid text restores the spinbox's own text colour and clears that knob's message.

### Applying a change

- A valid change from a spinbox or a preset sets `App.model` and applies it to the running layout: `self.layout = self.layout.with_model(model)`, then `self.wake()`.
- Positions, the iteration count and the jitter schedule carry on. The status-line energy jumps, because the energy function changed.
- `new_graph` creates layouts with `App.model`, and Randomize keeps it through `randomized_layout`. The app starts on `BALLOON`.
- After every change the preset menu shows `preset_name(App.model)`.

### Functions testable without a display

In `app.py`:

- `model_with_knob(model, field, text)` returns `(new model, None)` when `text` parses as a number that keeps the constraint of `field`, else `(None, constraint message)`. Text that is not a number is turned into `nan`, so the constraint message comes from `PowerLaw` itself and the UI and the model share one rule.
- `preset_name(model)` returns "Balloon (1/d)", "Dense (1/d²)" or "Custom".
- `PRESETS` lists the `(name, model)` pairs for the option menu.

## Tests

`tests/test_layout.py`:

- `PowerLaw` accepts its boundary values (`strength=0`, a small positive exponent) and rejects `strength < 0`, `exponent <= 0`, `nan` and infinity, with messages naming the knob.
- `BALLOON` reproduces the current energies and forces. The existing energy tests stay as they are and run on the default model.
- `DENSE`: two unconnected nodes at distance 5 have energy `1/5 - 1` (at `p = 2` the energy is `c/d - c`) and push each other apart with `1/25`.
- The energy is continuous across `p = 1`: exponents `1 ± 1e-6` give energies within `1e-4` of `p = 1`.
- `delta` is the negative gradient of the energy for `p` in `{0.5, 1, 1.5, 2, 3}`, extending the current gradient test.
- All functions that derive a new layout keep the model.
- `test_improved_takes_a_step_that_lowers_the_energy_but_raises_the_tension` uses `BALLOON`, so its layout stays valid.

`tests/test_app.py`:

- `model_with_knob` sets a field from text with spaces around a number, and keeps the other field.
- `model_with_knob` rejects `''`, `'abc'`, `'-1'` for strength, `'0'` for exponent and `'nan'`, returning the constraint message.
- `preset_name` returns the preset names for `BALLOON` and `DENSE`, and "Custom" for any other model.

Not tested without a display: how the widgets look and act (red text, the message row, the option menu). The user checks these in the app.

## Follow-up: stress

Not part of this task. Stress puts a spring between every pair, with a rest length of hop count × `EDGE_LENGTH`. It needs:

- all-pairs hop counts, computed once per graph;
- a model that also replaces the edge springs;
- a rule for pairs in different components.

It will be added as a third preset-menu entry with its own knobs.
