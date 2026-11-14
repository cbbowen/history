# Changelog

Breaking changes are marked **Breaking**. Versions before 0.6.0 are not recorded here.

## 0.6.0 (unreleased)

### Changed

- **Breaking**: the defaults for `Action::Context`, `Action::Error` and `Action::Centralizer` now
  need the nightly-only `associated_type_defaults` cargo feature. Without it the crate builds on
  stable, and every implementation has to name all three types.

### Fixed

- `pop_actions(0)` now does nothing. It used to rebuild whichever cache windows earlier pushes
  had left empty, so a call that removed nothing could apply actions and, for a fallible action,
  fail.
- `remove_action` and `pop_actions` no longer apply actions twice while rebuilding the cache.
  The most recent state was replayed from the newest state the operation kept rather than from
  the nearer ones it had just rebuilt, which doubled the applications of a removal with nothing
  commuting and added up to half again to a bulk pop.
