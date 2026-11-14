# Changelog

Breaking changes are marked **Breaking**. Versions before 0.6.0 are not recorded here.

## 0.6.0 (unreleased)

### Added

- `History::push_actions`, with the usual `try_` and `_with` variants, which adds several
  actions at once, and an `Extend` implementation that does the same. Each action is still a
  version of its own, but the state is cloned *O*(log *k*) times for *k* actions, where pushing
  them one at a time clones it *k* times. A failed `try_push_actions` adds none of them and
  returns them all in a `PushActionsError`.
- `History::action_applied_at`, which returns the action applied at a version, and
  `History::versions_and_actions`, which returns every action alongside the version it is
  applied at. That version is how a removal names the action to remove.

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
- **Breaking**: the `remove_action` family is now the `remove_action_applied_at` family, and
  takes the `Version` the action is applied at where it took a bare index. The index counted
  from the first action ever pushed, so after `forget_actions` it was no longer the action's
  position in `History::actions`, and looking an action up by position removed a different one.
  The number is the same: the action that had index `i` is the one applied at version `i`,
  which `versions_and_actions` returns with it, and for the oldest action that is
  `initial_version()`. `RemoveActionError::IndexOutOfRange` and `IndexForgotten`, which held the
  index, are `VersionOutOfRange` and `VersionForgotten`, which hold the version.
- **Breaking**: a failed removal now leaves the history exactly as it was, like every other
  operation. It used to leave the action moved past some of the actions it commutes with.
  `RemoveActionError::ActionFailed` is accordingly a tuple variant holding only the error, no
  longer a struct variant that also carried the action's new index.
- Removing an action applies fewer actions. Removing one that commutes with everything after it
  applies none of them, and reporting that actions commute never costs an application over
  reporting that nothing does, which it used to.
- A removal now relies only on the property `Centralizer` documents. A failure part of the way
  through used to leave states that were right only if the actions also commuted in the
  everyday sense, which that property does not imply.
- `forget_actions` now takes time proportional to the actions it leaves, as documented. It
  copied every action it folded in to a new vector as well. The vector it returns can have room
  for more than it contains.
- **Breaking**: the defaults for `Action::Context`, `Action::Error` and `Action::Centralizer` now
  need the nightly-only `associated_type_defaults` cargo feature. Without it the crate builds on
  stable, and every implementation has to name all three types.
- The documentation no longer describes how the history is implemented. It says what each
  operation does and what each trait requires, and gives time complexity as asymptotic bounds
  only, amortized and worst case, with what they count stated once on `History`.

### Fixed

- The documentation of `Version`, which said that a version is the index that removes the
  action that produced it. A removal names the action applied *at* a version; the action that
  produced the version is the one before. It also now says that pops and removals reuse and
  shift version numbers, where it said that versions count every action ever applied.
- The documented cost of a removal, which was said to take *O*(log *n*) applications for an
  action that commutes with the actions after it. It is *O*(*m* + log *n*) amortized for *m*
  actions after the one removed, and *O*(*n*) in the worst case.
- The documented cost of a pop, which said that most pops apply nothing. It is *O*(log *n*)
  amortized and *O*(*n*) in the worst case.
- `Action::apply_batch` is no longer called with an empty slice, and the requirements on it and
  on `Action::apply` are now written down: applying an action must give a state that depends
  only on the action and the state it is applied to, and must be safe to repeat.
- The documentation of the defaults for `Action`'s associated types, which promised them to
  builds that do not have the `associated_type_defaults` feature. The feature is now mentioned
  in the README.
- The description of what `forget_actions` bounds, in its documentation and the README. It is
  the number of actions left that is within a factor of what was asked for, not the number
  folded in, which can be none.
- `pop_actions(0)` now does nothing. It could apply actions and, for a fallible action, fail.
- A removal and `pop_actions` no longer apply actions twice. That doubled the applications of a
  removal with nothing commuting and, for a bulk pop, added half again on average and nearly as
  much again at worst.
