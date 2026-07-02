# GLASSBOX — v8: 1.0, the endgame

Status: vision doc, and the last one. There is no v9 planned, on purpose.

v3 polished inspection. v4 added discovery and intervention. v5 made it science. v6 made it an
instrument. v7 widened what it points at. v8 adds **no new capabilities at all** — it is the
version where GLASSBOX stops growing and starts being *finished*. A roadmap that never ends is
a project that never ships; this is the point.

---

## The 1.0 release

### 1. The cull
Run the usage question over every tab, mode, and toggle shipped since v3: what did nobody use?
Delete it. Every `ponytail:` debt comment gets resolved — either promoted to real (the ceiling
was hit) or the comment deleted (it never was). The v6 circuit-CI suite is the safety net that
makes deletion cheap. Target: 1.0 is *smaller* than v7.

### 2. Freeze the surfaces
- The experiment-spec format (v6-A1) becomes versioned and stable — it's the contract other
  people's repos depend on.
- The probe protocol (v7-B1) becomes the documented extension point — the *only* one.
- Everything else is explicitly internal and free to change.

### 3. Write it down
The docs GLASSBOX actually owes the world, in order:
- **The course**: v4's guided tours polished into a start-to-finish "intro to mech interp by
  doing" — the lens → heads → ablation → patching → features → circuits → grokking arc.
  This is the artifact with the longest half-life; treat it as the flagship deliverable.
- **The instrument paper/post**: what GLASSBOX measures, its faithfulness bounds (v6-C1's
  numbers), and the circuit-CI idea. One good write-up outlives ten features.
- **Maintainer docs**: how to add a model, a probe, an experiment. Thin, accurate, tested.

### 4. Hand it off
- Findings library (v5-E1) and experiment repos are the community mechanism; 1.0 makes
  "contribute a finding" a documented 10-minute path.
- Define the maintenance contract honestly: dependency bumps gated by the circuit CI, new
  models by registry PR + green faithfulness test, new methods → someone forks or the probe
  protocol carries it. The maintainer's job shrinks to gatekeeping invariants.

### 5. The invariants, one last time
The things v8 hands to whoever comes next, unchanged since the weekend-2 faithfulness spike
(plans/GLASSBOX-weekend2-plan.md) because they were measured, not guessed:
- Decode through the model's own final norm; never trust a convenience helper unverified.
- Correctness beats speed; a device or method that's silently wrong is worthless.
- Every numerics path has a faithfulness test.
- Data (registries, specs, findings) over behavior; behavior over abstraction.
- Build the smallest thing that answers the question, then stop.

---

## After 1.0

Maintenance mode is not death — it's the tool doing its job without demanding attention.
New interp methods will keep appearing; most belong in forks, probes, or new projects that
GLASSBOX's write-ups helped someone start. If a future capability truly demands a v9, it will
justify itself against this doc's cull list — and that argument is the feature bar working as
intended.

The best roadmap ends. This one ends here.
