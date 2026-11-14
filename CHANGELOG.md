# Changelog

Breaking changes are marked **Breaking**. Versions before 0.6.0 are not recorded here.

## 0.6.0 (unreleased)

### Added

- `History::push_actions`, with the usual `try_` and `_with` variants, which adds several
  actions at once, and an `Extend` implementation that does the same. Each action is still a
  version of its own, but the state is cloned once for each state the push will keep cached,
  *O*(log *k*) for *k* actions, where pushing them one at a time clones it *k* times. A failed
  `try_push_actions` adds none of them and returns them all in a `PushActionsError`.

### Changed

- **Breaking**: `Action::inverse` is now `Centralizer::inverse`, and it is a required method.
  The provided implementation restored the whole previous state, which is only right for an
  action that commutes with nothing, so a centralizer that reported commuting actions without
  also overriding `inverse` silently produced wrong states. The two can no longer be written
  apart. To migrate, move the body of `inverse` into the `Centralizer` implementation, reading
  what it needs of the action from the centralizer; an action that uses `NonCommutative` and
  did not override `inverse` needs no change.
- **Breaking**: a failed `try_remove_action` now leaves the history exactly as it was, like
  every other operation. It used to leave the action moved past some of the actions it commutes
  with. `RemoveActionError::ActionFailed` is accordingly a tuple variant holding only the error,
  no longer a struct variant that also carried the action's new index.
- Removing an action no longer applies any of the actions it commutes with. The cached states
  among them are corrected with `Centralizer::inverse` alone, where each used to cost an
  application as well, and the state the action was applied to is only rebuilt when one of them
  needs it. Reporting that actions commute now never costs an application over reporting that
  nothing does; it used to cost more whenever the commuting run ended before the next cached
  state.
- A removal now relies only on the property `Centralizer` documents. A failure part of the way
  through used to leave cached states that were right only if the actions also commuted in the
  everyday sense, which that property does not imply.
- **Breaking**: the defaults for `Action::Context`, `Action::Error` and `Action::Centralizer` now
  need the nightly-only `associated_type_defaults` cargo feature. Without it the crate builds on
  stable, and every implementation has to name all three types.

### Fixed

- The documented cost of the `remove_action` family. Removing an action that commutes with the
  actions after it was said to take *O*(log *n*) applications; reconstructing the state the
  action was applied to makes it linear in the distance to the end, and the bound is amortized
  rather than per call because a removal also repairs the cache as a pop does.
- `Action::apply_batch` is no longer called with an empty slice.
- `pop_actions(0)` now does nothing. It used to rebuild whichever cache windows earlier pushes
  had left empty, so a call that removed nothing could apply actions and, for a fallible action,
  fail.
- `remove_action` and `pop_actions` no longer apply actions twice while rebuilding the cache.
  The most recent state was replayed from the newest state the operation kept rather than from
  the nearer ones it had just rebuilt, which doubled the applications of a removal with nothing
  commuting and added up to half again to a bulk pop.
