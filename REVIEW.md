# Code review — `src/lib.rs`

A review of the crate at `0.3.0`, after the geometric-window cache landed. Findings are ordered
by what to fix first. Each is marked with its disposition.

The cache design itself holds up. `prune_cache` (each window run always reduces to exactly its
first and last member, so occupancy stays in `1..=2`), `plan_refill`'s coarse-to-fine ordering and
midpoint placement, `shift_late`'s rotate-and-repair loop (each iteration leaves the history fully
consistent, so the documented failure semantics are accurate), and the index remapping in
`try_remove_action_with`'s replay closure — including the subtlety that `index` is rebound to
`shift_late`'s return so the repaired cache entries at versions in `(old_index, target]` fall
inside `keep` rather than being discarded — were all traced and no bug was found in any of them.

## Logic / correctness

### 1. `try_get_state_with` bypasses `Action::apply_batch` — *fixed*

```rust
self.actions[recent_version.0..version.0]
    .iter()
    .try_fold(state.clone(), |s, a| a.apply(s, context))
```

Every other replay path goes through `A::apply_batch` (`plan_refill`'s closures in pop,
`pop_actions`, and `remove`). This one did not, so an implementor who overrides `apply_batch` to
fuse a run of actions got that optimization on pops and removals but not on `get_state` — and not
on `shift_late`, which reaches `state_at_index` through this function.

### 2. Six doctests documented one function but exercised another — *fixed*

They called the non-`try_` / non-`_with` sibling, so the documented method was never type-checked
or run:

| Doc on | Called instead |
| --- | --- |
| `try_pop_state_with` | `try_pop_action_with`, asserting `Some(Add(42))` rather than `Some(42)` |
| `pop_state_with` | `pop_state()` |
| `try_push_action` | `push_action()` |
| `try_pop_action_and_state` | `pop_action_and_state()` |
| `try_pop_action` | `pop_action()` |
| `try_pop_state` | `pop_state()` |

They compiled only because the example `Add` happens to satisfy both impl blocks' bounds. See
finding 13: the duplication is what produced them, so the fix is to generate the examples.

### 3. The `Centralizer` contract was understated and mistyped — *fixed*

It read `a.inverse(b.apply(a.apply(s)))`, but `inverse` takes two arguments and returns nothing.
More importantly, `shift_late` calls `self.actions[position].inverse(&state_at_index, &mut shifted)`
where `shifted` is the state after a *run* of commuting actions and `state_at_index` is the state
from before any of them. What must actually hold is the sequence form: for any `b₁…bₖ` all
reported as commuting, inverting `a` out of `bₖ(…b₁(a(s)))` against `s` must give `bₖ(…b₁(s))`.
An implementor reading the single-action statement could write a conforming `inverse` that still
breaks `remove_action`.

### 4. `PushError` and `PopError` were unusable as errors — *fixed, except the cause chain*

`PushError` held `action` and `error` as private fields with no accessors and no `into_action()`,
so a failed push destroyed the caller's action. `PopError` was a private-field tuple struct.

Not fixed: none of the error types marks its inner error as a `source`, so the cause chain is
empty. Doing so needs `Action::Error: std::error::Error + 'static`, which is an API-breaking bound
on the public trait (it would reject `type Error = ()`, as used in the tests). That is a design
call for the crate owner, not a cleanup. Recommended if a `0.4` is on the table: the bound is
cheap — `Infallible` already satisfies it — and it buys `#[source]` plus a `Display` that actually
says what went wrong instead of the current constant string.

### 5. `Version`'s field was private with no accessor — *fixed*

`versions()` handed out `Version`, but `try_remove_action` wanted a `usize` index and
`RemoveActionError::IndexOutOfRange` reported one, with no supported way to bridge them.

### 6. Derived bounds on the error types were too strong — *fixed*

`#[derive(Debug, PartialEq, Eq)]` on `PopError<A>`, `GetStateError<A>`, and `RemoveActionError<A>`
generated `A: Debug` / `A: PartialEq` bounds even though none of them stores an `A` — only
`A::Error`. The consequence: `history.try_pop_action().unwrap()` would not compile unless the
*action* type was `Debug`.

## Performance

### 7. `prune_cache` self-swapped a full state per surviving entry — *fixed*

```rust
self.states.swap(write, read);
```

`slice::swap` has no `a == b` fast path; it lowers to `ptr::swap`, which unconditionally performs
three copies of `size_of::<(Version, A::State)>()`. Every push therefore paid *O*(log *n*) full
state copies through `prune_cache` even in the common case where nothing is dropped and
`write == read` throughout. For an inline `A::State` — an array, a fixed-size struct — that was
the dominant cost of a push.

### 8. `window_of` was computed twice per entry in `prune_cache` — *fixed*

`next` for `read + 1` was recomputed as `scale` on the following iteration, doubling the `ilog2`
count on the push hot path.

### 9. `plan_refill`'s final sort was removable — *fixed*

The most-recent-state entry pushed before the window loop is inert during it: its distance is 0 so
`occupies` is never true for it, and base selection filters `version <= target` where
`target < new_len`, so it is never chosen as a replay base either. The loop's targets,
`new_len - ((3 << scale) >> 1)` over descending `scale`, are already strictly ascending. Computing
the most-recent state *after* the loop leaves `plan` sorted by construction.

### 10. `sort_by_key` in `install_refill` — *fixed*

Versions are unique so stability is meaningless, but the stable sort allocates a `len / 2` buffer
and moves whole `A::State` values through it. `sort_unstable_by_key` is a drop-in.

### 11. `plan_refill`'s occupancy and base lookups are *O*(log² *n*) — *not fixed, deliberately*

Both scans rescan `states.iter().chain(plan.iter())` per window. Both sequences are sorted and the
loop walks windows monotonically, so two cursors would make it *O*(log *n*).

Left alone on purpose. The term is dominated by the *O*(2^scale) replays the loop schedules, log
*n* is at most 64, and the change would add monotone index bookkeeping to the most delicate
function in the crate — the occupancy cursor and the base cursor advance on different bounds
(`new_len - 2^(scale+1)` versus `new_len - 3·2^(scale-1)`), so they cannot share one index. That is
a poor risk-to-benefit trade against code that is currently obviously correct.

## Cleanups — all fixed

- `self.actions.len() - k.min(self.actions.len())` → `saturating_sub(k)`.
- `History::new`: `[(Version::default(), initial)].into_iter().collect()` → `vec![(Version(0), initial)]`.
- `get_cached_state` and `get_recent_state_index` were each called once, from `try_get_state_with`,
  and carried two `unwrap()`s between them; `get_recent_state_index` returned a `Result` that could
  only ever hold one variant. Inlined.
- `cached_versions`: `.map(|(version, _)| version).cloned()` → `.map(|&(version, _)| version)`.
- `try_pop_action_and_state_with` was `try_pop_actions_with` with `keep = states.len() - 1`
  hardcoded in place of the `partition_point` (they are provably equal) plus the returned state.
  Folded into one private `pop_to`.
- Roughly 950 of the file's ~1280 non-test lines were near-verbatim duplicated doc comments across
  the four impl blocks. That duplication is exactly what produced finding 2, so the example bodies
  and complexity sections are now generated from one definition each.
- `let before = *history.last_state();` was dead in ~15 doctests, and `let version = …` in two more.
- Clippy's two `clone_on_copy` warnings in the tests.
- `Action::Context`, `Action::Error`, `Action::Centralizer`, and both `Centralizer` methods had no
  doc comments. `#![warn(missing_docs)]` now catches these and future ones.
- `try_pop_actions` / `pop_actions` were the only public operations without a doctest.
