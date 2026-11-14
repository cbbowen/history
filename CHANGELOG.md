# Changelog

Breaking changes are marked **Breaking**. Versions before 0.6.0 are not recorded here.

## 0.6.0 (unreleased)

### Added

- `History::push_actions`, with the usual `try_` and `_with` variants, which adds several
  actions at once, and an `Extend` implementation that does the same. Each action is still a
  version of its own, but the state is cloned once for each state the push will keep cached,
  *O*(log *k*) for *k* actions, where pushing them one at a time clones it *k* times. A failed
  `try_push_actions` adds none of them and returns them all in a `PushActionsError`.
- `History::action`, which returns the action at an index. The indices are the ones the
  `remove_action` family takes: they count from the first action ever pushed, so after
  `forget_actions` they are no longer positions in `History::actions`, and looking an action up
  by position removed a different one.

### Changed

- **Breaking**: `Action::inverse` is now `Centralizer::inverse`, and it is a required method.
  The provided implementation restored the whole previous state, which is only right for an
  action that commutes with nothing, so a centralizer that reported commuting actions without
  also overriding `inverse` silently produced wrong states. The two can no longer be written
  apart. To migrate, move the body of `inverse` into the `Centralizer` implementation, reading
  what it needs of the action from the centralizer, which may mean keeping more of the action
  in it: `for_action` is the only place that sees the action, and a centralizer for an action
  type with parameters of its own generally has to copy what it needs rather than borrow it. An
  action that uses `NonCommutative` needs no change if it did not override `inverse`, and
  should just delete the override if it did: it was never called.
- **Breaking**: a failed `try_remove_action` now leaves the history exactly as it was, like
  every other operation. It used to leave the action moved past some of the actions it commutes
  with. `RemoveActionError::ActionFailed` is accordingly a tuple variant holding only the error,
  no longer a struct variant that also carried the action's new index.
- Removing an action no longer applies an action to correct a cached state. The cached states
  among the actions it commutes with are corrected with `Centralizer::inverse` alone, where each
  used to cost an application as well, and the state the action was applied to is only rebuilt
  when one of them needs it. None of the actions before the last corrected state is applied
  again, so removing an action that commutes with everything after it applies none of them.
  Reporting that actions commute now never costs an application over reporting that nothing
  does; it used to cost more whenever the commuting run ended before the next cached state.
- `forget_actions` now takes time proportional to the actions it leaves, as documented. It
  copied every action it folded in to a new vector as well. The vector it returns is now the
  buffer the history held, so it can have room for more than it contains.
- A removal now relies only on the property `Centralizer` documents. A failure part of the way
  through used to leave cached states that were right only if the actions also commuted in the
  everyday sense, which that property does not imply.
- **Breaking**: the defaults for `Action::Context`, `Action::Error` and `Action::Centralizer` now
  need the nightly-only `associated_type_defaults` cargo feature. Without it the crate builds on
  stable, and every implementation has to name all three types.

- The time complexity sections now give asymptotic bounds only, amortized and worst case, with
  what they count stated once on `History`. They no longer describe how many applications or
  clones an operation happens to perform, or when the cache makes one cheaper.

### Fixed

- The documentation of `Version`, which said that a version is the index that removes the
  action that produced it. It is the index of the action applied *to* it; the action that
  produced it is one earlier. It also now says that pops and removals reuse and shift version
  numbers, where it said that versions count every action ever applied.
- The documented cost of the `remove_action` family. Removing an action that commutes with the
  actions after it was said to take *O*(log *n*) applications; reconstructing the state the
  action was applied to makes it linear in the distance to the end, and the bound is amortized
  rather than per call because a removal also repairs the cache as a pop does. It is now given
  as *O*(*m* + log *n*) amortized for *m* actions after the index, and *O*(*n*) in the worst
  case.
- `Action::apply_batch` is no longer called with an empty slice, and the requirements on it and
  on `Action::apply` are now written down: applying an action must give a state that depends
  only on the action and the state it is applied to, and must be safe to repeat, because the
  history applies actions again whenever it rebuilds a state.
- The documentation of the defaults for `Action`'s associated types, which promised them to
  builds that do not have the `associated_type_defaults` feature. The feature is now mentioned
  in the README.
- The documented cost of a pop, which said that most pops replay nothing. That is true only of
  a pop that undoes the push before it, and then only in a history of single pushes and pops; in
  a run of pops most replay at least one action. The documentation now gives the bounds alone:
  *O*(log *n*) amortized and *O*(*n*) in the worst case.
- The description of what `forget_actions` bounds, in its documentation and the README. It is
  the number of actions left that is within a factor of what was asked for, not the number
  folded in, which can be none.
- `pop_actions(0)` now does nothing. It used to rebuild whichever cache windows earlier pushes
  had left empty, so a call that removed nothing could apply actions and, for a fallible action,
  fail.
- `remove_action` and `pop_actions` no longer apply actions twice while rebuilding the cache.
  The most recent state was replayed from the newest state the operation kept rather than from
  the nearer ones it had just rebuilt, which doubled the applications of a removal with nothing
  commuting and, for a bulk pop, added half again on average and nearly as much again at worst.
