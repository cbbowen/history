#![feature(associated_type_defaults)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

/// An action which can be added to a [`History`].
pub trait Action: Sized {
    /// The type of state this action affects.
    type State: Clone;

    /// Anything applying an action needs that the history does not hold, such as a resource
    /// pool or an interner. Every operation that can apply an action takes one by `&mut`.
    ///
    /// Defaults to `()`, which unlocks the shorter methods that do not take one at all.
    type Context = ();

    /// The reason applying this action can fail.
    ///
    /// Defaults to [`Infallible`](std::convert::Infallible), which unlocks the methods that
    /// return the result directly rather than a [`Result`].
    ///
    /// This crate's error types report a value of this type as their
    /// [`source`](std::error::Error::source), so their own [`Display`](std::fmt::Display) says
    /// only which operation failed and leaves the reason to the chain.
    type Error: std::error::Error + 'static = std::convert::Infallible;

    /// Which other actions this one commutes with, which is what lets the `remove_action` family
    /// avoid replaying the actions after the one it removes.
    ///
    /// Defaults to [`NonCommutative`], which never reports that anything commutes: always sound,
    /// and never faster.
    type Centralizer<'a>: Centralizer<'a, Self> = NonCommutative;

    /// Applies this action to a state, producing a new state.
    fn apply(
        &self,
        state: Self::State,
        context: &mut Self::Context,
    ) -> Result<Self::State, Self::Error>;

    /// Applies a sequence of actions to a state, producing a new state.
    fn apply_batch(
        actions: &[Self],
        mut state: Self::State,
        context: &mut Self::Context,
    ) -> Result<Self::State, Self::Error> {
        for action in actions {
            state = action.apply(state, context)?;
        }
        Ok(state)
    }

    /// Applies the inverse of this action to a given state in place.
    ///
    /// For all `previous_state`, `self.inverse(previous_state, state)` where `state` is
    /// `self.apply(previous_state)` must leave `state` equivalent to `previous_state`. Observe that
    /// if [`Self::Centralizer`] never reports that an action commutes, cloning `previous_state`
    /// into `state` is always a correct implementation. However, history surgery can be made more
    /// efficient with an implementation that only restores the portion of the state this action
    /// actually affects.
    ///
    /// If [`Self::Centralizer`] *does* report that actions commute, this obligation is not
    /// enough on its own: `state` is then a state some commuting actions further on, and
    /// [`Centralizer`] documents the stronger property that must hold.
    fn inverse(&self, previous_state: &Self::State, state: &mut Self::State) {
        state.clone_from(previous_state);
    }
}

/// Identifies a specific version in the history.
///
/// Version `i` is the state reached by applying the first `i` actions, so it is also the index of
/// the action that produced it plus one, and the index the `remove_action` family takes to remove
/// that action.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(usize);

impl Version {
    /// Returns the number of actions applied to reach this version.
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<Version> for usize {
    fn from(version: Version) -> Self {
        version.0
    }
}

/// Represents the set of actions an action commutes with.
///
/// To be precise, let `c` be `Centralizer::for_action(a)` and let `b_1, ..., b_k` be any actions
/// with `c.commutes(b_i)` for every `i`. Write `bs(s)` for applying them in order to a state `s`.
/// Then for all `s`, inverting `a` back out of the state reached by applying it and then them,
///
/// ```text
/// let mut state = bs(a.apply(s));
/// a.inverse(&s, &mut state);
/// ```
///
/// *must* leave `state` equivalent to `bs(s)`. Note that `a.inverse` is passed the state from
/// before `a` was applied, not the one immediately preceding `state`; the two differ by exactly
/// `b_1, ..., b_k`, which is what commuting has to make irrelevant. The single-action case
/// `k = 1` is not sufficient: the `remove_action` family shifts an action past a whole run of
/// commuting actions at once.
///
/// False negative results will make the `remove_action` family of functions fall back to slower
/// (but correct) implementations. False positive results will produce incorrect states.
pub trait Centralizer<'a, A: Action> {
    /// Returns the centralizer of `action`.
    fn for_action(action: &'a A) -> Self;

    /// Returns whether the action this centralizer was built for commutes with `other`.
    ///
    /// Returning `false` is always sound; see the trait documentation for what returning `true`
    /// promises.
    fn commutes(&self, other: &A) -> bool;
}

/// A centralizer for an action that might not commute with any other action.
pub struct NonCommutative;

impl<'a, A: Action> Centralizer<'a, A> for NonCommutative {
    fn for_action(_action: &'a A) -> Self {
        Self
    }
    fn commutes(&self, _other: &A) -> bool {
        false
    }
}

// These error types are written out rather than derived because every derive would bound the
// action type itself, which none of them stores except `PushError`. Deriving `Debug` on
// `PopError<A>` would make `history.try_pop_action().unwrap()` require `A: Debug`.

/// The error type of [`History::try_push_action`] and [`History::try_push_action_with`].
///
/// A failed push does not add the action to the history, so the error carries it back to the
/// caller.
pub struct PushError<A: Action> {
    action: A,
    error: A::Error,
}

impl<A: Action> PushError<A> {
    /// Returns the action that could not be applied.
    pub fn action(&self) -> &A {
        &self.action
    }

    /// Returns the error that applying the action produced.
    pub fn error(&self) -> &A::Error {
        &self.error
    }

    /// Returns the action that could not be applied, consuming the error.
    pub fn into_action(self) -> A {
        self.action
    }

    /// Returns the error that applying the action produced, consuming the rest.
    pub fn into_error(self) -> A::Error {
        self.error
    }

    /// Splits into the action that could not be applied and the error it produced.
    pub fn into_parts(self) -> (A, A::Error) {
        (self.action, self.error)
    }
}

impl<A: Action> std::fmt::Display for PushError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to apply action")
    }
}

impl<A: Action + std::fmt::Debug> std::fmt::Debug for PushError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushError")
            .field("action", &self.action)
            .field("error", &self.error)
            .finish()
    }
}

impl<A: Action + PartialEq> PartialEq for PushError<A>
where
    A::Error: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.action == other.action && self.error == other.error
    }
}

impl<A: Action + Eq> Eq for PushError<A> where A::Error: Eq {}

impl<A: Action + std::fmt::Debug> std::error::Error for PushError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// The error type of [`History::try_get_state`] and [`History::try_get_state_with`].
pub enum GetStateError<A: Action> {
    /// The requested version is later than the most recent one.
    VersionOutOfRange(Version),

    /// Applying an action failed while reconstructing the state.
    ActionFailed(A::Error),
}

impl<A: Action> std::fmt::Display for GetStateError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionOutOfRange(_) => f.write_str("version out of range"),
            Self::ActionFailed(_) => f.write_str("failed to apply action"),
        }
    }
}

impl<A: Action> std::fmt::Debug for GetStateError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionOutOfRange(version) => {
                f.debug_tuple("VersionOutOfRange").field(version).finish()
            }
            Self::ActionFailed(error) => f.debug_tuple("ActionFailed").field(error).finish(),
        }
    }
}

impl<A: Action> PartialEq for GetStateError<A>
where
    A::Error: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::VersionOutOfRange(a), Self::VersionOutOfRange(b)) => a == b,
            (Self::ActionFailed(a), Self::ActionFailed(b)) => a == b,
            _ => false,
        }
    }
}

impl<A: Action> Eq for GetStateError<A> where A::Error: Eq {}

impl<A: Action> std::error::Error for GetStateError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::VersionOutOfRange(_) => None,
            Self::ActionFailed(error) => Some(error),
        }
    }
}

/// The error type of [`History::try_remove_action`] and [`History::try_remove_action_with`].
pub enum RemoveActionError<A: Action> {
    /// No action has the requested index.
    IndexOutOfRange(usize),

    /// Applying an action failed while rebuilding the cached states. The action to remove is
    /// still in the history at `index`, which may differ from the requested index: the action
    /// may have been shifted past actions it commutes with.
    ActionFailed {
        /// Where the action to remove now sits.
        index: usize,
        /// The error that applying an action produced.
        error: A::Error,
    },
}

impl<A: Action> std::fmt::Display for RemoveActionError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfRange(_) => f.write_str("action index out of range"),
            Self::ActionFailed { .. } => f.write_str("failed to apply action"),
        }
    }
}

impl<A: Action> std::fmt::Debug for RemoveActionError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfRange(index) => f.debug_tuple("IndexOutOfRange").field(index).finish(),
            Self::ActionFailed { index, error } => f
                .debug_struct("ActionFailed")
                .field("index", index)
                .field("error", error)
                .finish(),
        }
    }
}

impl<A: Action> PartialEq for RemoveActionError<A>
where
    A::Error: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::IndexOutOfRange(a), Self::IndexOutOfRange(b)) => a == b,
            (
                Self::ActionFailed { index, error },
                Self::ActionFailed {
                    index: other_index,
                    error: other_error,
                },
            ) => index == other_index && error == other_error,
            _ => false,
        }
    }
}

impl<A: Action> Eq for RemoveActionError<A> where A::Error: Eq {}

impl<A: Action> std::error::Error for RemoveActionError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IndexOutOfRange(_) => None,
            Self::ActionFailed { error, .. } => Some(error),
        }
    }
}

/// The error type of [`History::try_pop_action_and_state`], [`History::try_pop_action`],
/// [`History::try_pop_state`], and [`History::try_pop_actions`].
pub struct PopError<A: Action>(A::Error);

impl<A: Action> PopError<A> {
    /// Returns the error that applying an action produced.
    pub fn error(&self) -> &A::Error {
        &self.0
    }

    /// Returns the error that applying an action produced, consuming this.
    pub fn into_error(self) -> A::Error {
        self.0
    }
}

impl<A: Action> std::fmt::Display for PopError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to apply action")
    }
}

impl<A: Action> std::fmt::Debug for PopError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PopError").field(&self.0).finish()
    }
}

impl<A: Action> PartialEq for PopError<A>
where
    A::Error: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<A: Action> Eq for PopError<A> where A::Error: Eq {}

impl<A: Action> std::error::Error for PopError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

// What shortening a history removed.
struct Popped<A: Action> {
    // The removed actions, most recent first.
    actions: Vec<A>,
    // The state the most recent removed action produced, absent only when nothing was removed.
    state: Option<A::State>,
}

/// The history of state as actions are applied to it.
#[derive(Debug, Clone)]
pub struct History<A: Action> {
    actions: Vec<A>,
    states: Vec<(Version, A::State)>,
}

impl<A: Action> Default for History<A>
where
    A::State: Default,
{
    fn default() -> Self {
        Self::new(Default::default())
    }
}

// Every operation comes in four variants -- fallible or not, taking a context or not -- whose
// documentation differs only in the call it demonstrates. Written out, the four drifted: six of
// the examples ended up calling a *different* variant than the one they documented, and still
// passed, because the example action satisfies every variant's bounds. Generating the bulky parts
// from one definition each keeps them in step.
//
// The literals below start at column zero because a doc attribute's contents are taken verbatim,
// and leading indentation inside a fenced block would become part of the code.

// The `# Example` section for the `push_action` family. `call` must evaluate to the new version.
macro_rules! push_example {
    ($setup:expr, $call:expr) => {
        concat!(
            "# Example

```
# use history::*;
# #[derive(Debug, PartialEq, Eq)]
# struct Add(i32);
# impl Action for Add {
#  type State = i32;
#  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
# }
# let mut history = History::default();
",
            $setup,
            "assert_eq!(*history.last_state(), 0);
let version = ",
            $call,
            ";
assert_eq!(*history.last_state(), 42);
assert_eq!(history.last_version(), version);
```"
        )
    };
}

// The `# Example` section for the `pop_action` family. `call` must evaluate to an `Option`
// holding `popped`.
macro_rules! pop_example {
    ($setup:expr, $call:expr, $popped:expr) => {
        concat!(
            "# Example

```
# use history::*;
# #[derive(Debug, PartialEq, Eq)]
# struct Add(i32);
# impl Action for Add {
#  type State = i32;
#  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
# }
# let mut history = History::default();
",
            $setup,
            "history.push_action(Add(42));
assert_eq!(*history.last_state(), 42);
assert_eq!(",
            $call,
            ", Some(",
            $popped,
            "));
assert_eq!(*history.last_state(), 0);
```"
        )
    };
}

// The `# Example` section for the `pop_actions` family. `call` must remove the two most recent
// actions and evaluate to them.
macro_rules! pop_actions_example {
    ($setup:expr, $call:expr) => {
        concat!(
            "# Example

```
# use history::*;
# #[derive(Debug, PartialEq, Eq)]
# struct Add(i32);
# impl Action for Add {
#  type State = i32;
#  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
# }
# let mut history = History::default();
",
            $setup,
            "history.push_action(Add(1));
history.push_action(Add(2));
history.push_action(Add(3));
assert_eq!(*history.last_state(), 6);
// The actions come back most recent first.
assert_eq!(",
            $call,
            ", vec![Add(3), Add(2)]);
assert_eq!(*history.last_state(), 1);
```"
        )
    };
}

// The `# Example` section for the `remove_action` family. `call` removes the action at index 0
// and must evaluate to `first` the first time and `second` the second.
macro_rules! remove_example {
    ($setup:expr, $call:expr, $first:expr, $second:expr) => {
        concat!(
            "# Example

```
# use history::*;
# #[derive(Debug, Clone, Copy, PartialEq, Eq)]
# struct Set(usize, i32);
# struct OtherIndices(usize);
# impl<'a> Centralizer<'a, Set> for OtherIndices {
#  fn for_action(action: &'a Set) -> Self { Self(action.0) }
#  fn commutes(&self, other: &Set) -> bool { self.0 != other.0 }
# }
# impl Action for Set {
#  type State = Vec<i32>;
#  type Centralizer<'a> = OtherIndices;
#  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
#   state[self.0] = self.1;
#   Ok(state)
#  }
#  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
#   state[self.0] = previous_state[self.0];
#  }
# }
# let mut history = History::new(vec![0, 0]);
",
            $setup,
            "history.push_action(Set(0, 1));
history.push_action(Set(1, 2));
assert_eq!(*history.last_state(), vec![1, 2]);
// `Set(0, 1)` commutes with `Set(1, 2)`, so this removal is cheap.
assert_eq!(",
            $call,
            ", ",
            $first,
            ");
assert_eq!(*history.last_state(), vec![0, 2]);
// Removing an action that does not commute replays the actions after it.
history.push_action(Set(1, 4));
assert_eq!(",
            $call,
            ", ",
            $second,
            ");
assert_eq!(*history.last_state(), vec![0, 4]);
```"
        )
    };
}

macro_rules! push_complexity {
    () => {
        "# Time complexity

Performs exactly one application and one state clone, plus *O*(log *n*) bookkeeping."
    };
}

macro_rules! pop_complexity {
    () => {
        "# Time complexity

Takes *O*(log *n*) amortized time under any interleaving of pushes and pops. Most pops replay
nothing at all: the state they need is the one the push before them displaced."
    };
}

macro_rules! pop_actions_complexity {
    () => {
        "# Time complexity

Takes *O*(*k* (1 + log (*n*/*k*))) amortized time."
    };
}

macro_rules! remove_complexity {
    () => {
"# Time complexity

Takes *O*(*n*) time, where *n* is the number of actions after `index`. The consecutive actions
after `index` that the removed action commutes with are shifted past with only *O*(log *n*) state
clones, inversions, and applications; the actions after the first non-commuting one are re-applied
in full."
    };
}

// The hidden setup line the `_with` variants need, and the empty one the rest use.
macro_rules! context_setup {
    () => {
        "# let mut context = ();\n"
    };
}

impl<A: Action> History<A> {
    /// Constructs a new history with the given initial state.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # struct NoOp;
    /// # impl Action for NoOp {
    /// #  type State = i32;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(state) }
    /// # }
    /// let history = History::<NoOp>::new(42);
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    pub fn new(initial: A::State) -> Self {
        Self {
            actions: Vec::new(),
            states: vec![(Version(0), initial)],
        }
    }

    /// Returns all the actions from oldest to newest.
    pub fn actions(&self) -> impl Iterator<Item = &A> {
        self.actions.iter()
    }

    /// Returns all the actions from oldest to newest.
    pub fn into_actions(self) -> impl Iterator<Item = A> {
        self.actions.into_iter()
    }

    /// Returns all the versions from oldest to newest.
    pub fn versions(&self) -> impl Iterator<Item = Version> {
        (0..=self.actions.len()).map(Version)
    }

    // The cache holds states at geometrically spaced versions, dense near the most recent version
    // and sparse near the oldest. Rather than pinning each state to an exact version, it groups
    // them into *windows*: window `scale` covers the versions whose distance from the most recent
    // one lies in `2^scale..2^(scale + 1)`. Every window holds one or two states.
    //
    // The redundancy is what keeps stepping back and forth cheap. A layout pinned to the length
    // of the history is a binary counter, so alternating a push with a pop re-propagates the same
    // carry forever, rebuilding a state *O*(*n*) actions back on every cycle. Here a push only
    // ever drops states a window can spare and a pop only rebuilds a window that has run empty,
    // and a rebuilt state is placed in the middle of its window, so it survives another
    // *O*(2^scale) operations in either direction before it must be rebuilt again.

    // The number of windows a history of `len` actions has.
    fn window_count(len: usize) -> u32 {
        if len == 0 { 0 } else { len.ilog2() + 1 }
    }

    // The scale of the window holding the state at `version`, or `None` for the most recent
    // state, which is always cached and belongs to no window.
    fn window_of(Version(last): Version, Version(version): Version) -> Option<u32> {
        let distance = last - version;
        (distance > 0).then(|| distance.ilog2())
    }

    // Drops every cached state whose window still holds a state on either side of it, leaving the
    // nearest and furthest state of each window: under a push those two are, respectively, the
    // last to leave the window, and under a pop the same holds in reverse. Performs no action
    // applications and no state clones.
    fn prune_cache(&mut self) {
        let last = self.last_version();
        let scale_at = |states: &[(Version, A::State)], index: usize| {
            states
                .get(index)
                .and_then(|(version, _)| Self::window_of(last, *version))
        };

        let mut previous = None;
        let mut scale = scale_at(&self.states, 0);
        let mut write = 0;
        for read in 0..self.states.len() {
            // Carried from the previous iteration rather than recomputed, so each entry's window
            // is measured once. Reading ahead is sound because nothing past `read` has been
            // overwritten yet, and `previous` carries the scale the entry before `read` had
            // before it moved.
            let next = scale_at(&self.states, read + 1);
            let redundant = scale.is_some() && previous == scale && scale == next;
            previous = scale;
            scale = next;
            if !redundant {
                // Guarded because `swap` has no equal-index fast path: it copies the whole
                // element through a temporary either way, and states can be large.
                if write != read {
                    self.states.swap(write, read);
                }
                write += 1;
            }
        }
        self.states.truncate(write);
    }

    // Returns the states that must be added to `states` for it to cache a history of `new_len`
    // actions, ascending by version. `replay` must carry a state from one version of the *new*
    // history to a later one.
    //
    // Every action application a cache repair needs happens here, so a caller can run this before
    // touching the history and leave it untouched if an application fails.
    fn plan_refill(
        states: &[(Version, A::State)],
        new_len: usize,
        mut replay: impl FnMut(usize, usize, A::State, &mut A::Context) -> Result<A::State, A::Error>,
        context: &mut A::Context,
    ) -> Result<Vec<(Version, A::State)>, A::Error> {
        let mut plan: Vec<(Version, A::State)> = Vec::new();

        // Fill the windows from the coarsest to the finest so that each refill can replay from a
        // state an earlier iteration already planned. The coarsest window always holds the
        // initial state, so the body never runs for it and the midpoint below cannot underflow.
        // Successive targets ascend, which is what leaves `plan` sorted without a pass at the end.
        for scale in (0..Self::window_count(new_len)).rev() {
            let occupies =
                |(Version(version), _): &(Version, A::State)| (new_len - *version) >> scale == 1;
            if states.iter().any(occupies) || plan.iter().any(occupies) {
                continue;
            }
            let target = new_len - ((3 << scale) >> 1);
            let (Version(base), state) = states
                .iter()
                .chain(plan.iter())
                .filter(|(Version(version), _)| *version <= target)
                .max_by_key(|(version, _)| *version)
                .expect("the initial state is always cached");
            let state = replay(*base, target, state.clone(), context)?;
            plan.push((Version(target), state));
        }

        // The most recent state must always be cached: `last_state` hands it out directly. It is
        // planned last because it belongs to no window and sits at no version any refill above
        // replays from, so the loop neither sees it nor needs it -- and appending it keeps the
        // plan ascending.
        if states.last().map(|(version, _)| *version) != Some(Version(new_len)) {
            let (Version(base), state) = states.last().expect("the initial state is always cached");
            let state = replay(*base, new_len, state.clone(), context)?;
            plan.push((Version(new_len), state));
        }

        debug_assert!(plan.windows(2).all(|pair| pair[0].0 < pair[1].0));
        Ok(plan)
    }

    // Replaces the cached states after the first `keep` with `plan` and drops the ones the result
    // makes redundant. Must run after the actions have been updated, so that window scales are
    // measured against the new most recent version.
    fn install_refill(&mut self, keep: usize, plan: Vec<(Version, A::State)>) {
        self.states.truncate(keep);
        self.states.extend(plan);
        // Both halves are sorted, but a plan entry can fall before a kept one: a window is
        // refilled at its midpoint, and the kept states nearer the end sit in finer windows.
        // Unstable is fine and avoids a scratch buffer -- versions are unique, so the order is
        // fully determined.
        self.states.sort_unstable_by_key(|(version, _)| *version);
        self.prune_cache();
    }

    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// Returns an error carrying the action back if applying it fails; in that case, the history
    /// is unchanged.
    #[doc = push_example!(
        context_setup!(),
        "history.try_push_action_with(Add(42), &mut context).unwrap()"
    )]
    ///
    #[doc = push_complexity!()]
    pub fn try_push_action_with(
        &mut self,
        action: A,
        context: &mut A::Context,
    ) -> Result<Version, PushError<A>> {
        let new_state = match action.apply(self.last_state().clone(), context) {
            Ok(s) => s,
            Err(error) => return Err(PushError { action, error }),
        };
        self.actions.push(action);
        let new_version = Version(self.actions.len());
        self.states.push((new_version, new_state));

        // The state that was most recent lands in the finest window, so a push never leaves a
        // window empty and never has to replay anything; it only drops states a window can spare.
        self.prune_cache();
        Ok(new_version)
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!(
        context_setup!(),
        "history.try_pop_action_and_state_with(&mut context).unwrap()",
        "(Add(42), 42)"
    )]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_action_and_state_with(
        &mut self,
        context: &mut A::Context,
    ) -> Result<Option<(A, A::State)>, PopError<A>> {
        let Some(new_len) = self.actions.len().checked_sub(1) else {
            return Ok(None);
        };
        let Popped { mut actions, state } = self.pop_to(new_len, context)?;
        let action = actions.pop().expect("exactly one action was removed");
        let state = state.expect("the most recent state is always cached");
        Ok(Some((action, state)))
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!(
        context_setup!(),
        "history.try_pop_action_with(&mut context).unwrap()",
        "Add(42)"
    )]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_action_with(
        &mut self,
        context: &mut A::Context,
    ) -> Result<Option<A>, PopError<A>> {
        self.try_pop_action_and_state_with(context)
            .map(|o| o.map(|(a, _)| a))
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!(
        context_setup!(),
        "history.try_pop_state_with(&mut context).unwrap()",
        "42"
    )]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_state_with(
        &mut self,
        context: &mut A::Context,
    ) -> Result<Option<A::State>, PopError<A>> {
        self.try_pop_action_and_state_with(context)
            .map(|o| o.map(|(_, s)| s))
    }

    /// Removes the most recent `k` actions.
    ///
    /// Returns all actions that were removed, in reverse order, which may be fewer than `k` if
    /// there have been fewer than `k` actions. On error, the history is unchanged.
    #[doc = pop_actions_example!(
        context_setup!(),
        "history.try_pop_actions_with(2, &mut context).unwrap()"
    )]
    ///
    #[doc = pop_actions_complexity!()]
    pub fn try_pop_actions_with(
        &mut self,
        k: usize,
        context: &mut A::Context,
    ) -> Result<Vec<A>, PopError<A>> {
        let new_len = self.actions.len().saturating_sub(k);
        Ok(self.pop_to(new_len, context)?.actions)
    }

    // Shortens the history to `new_len` actions, returning them in reverse order along with the
    // state the most recent one produced, or `None` if nothing was removed.
    //
    // Every action application happens before any mutation, so on error the history is unchanged.
    fn pop_to(
        &mut self,
        new_len: usize,
        context: &mut A::Context,
    ) -> Result<Popped<A>, PopError<A>> {
        // Keep every cached state the shorter history can still use and rebuild only the windows
        // that running the end backwards leaves empty. When several actions go at once, that
        // skips the states the same number of successive pops would compute and then discard.
        let keep = self
            .states
            .partition_point(|(Version(version), _)| *version <= new_len);
        let plan = Self::plan_refill(
            &self.states[..keep],
            new_len,
            |from, to, state, context| A::apply_batch(&self.actions[from..to], state, context),
            context,
        )
        .map_err(PopError)?;

        // All mutation happens below and is infallible. The most recent state is always cached,
        // so if anything is coming off at all it is sitting at the end of the cache.
        let state = if keep < self.states.len() {
            let (version, state) = self.states.pop().expect("`keep` is a smaller index");
            debug_assert_eq!(version, self.last_version());
            Some(state)
        } else {
            None
        };
        let actions = self.actions.drain(new_len..).rev().collect();
        self.install_refill(keep, plan);
        Ok(Popped { actions, state })
    }

    /// Shifts the action at `index` past every consecutive later action it commutes with and
    /// returns its new index.
    ///
    /// On error, the action sits at the index carried by the error, and the history is valid.
    fn shift_late(
        &mut self,
        index: usize,
        context: &mut A::Context,
    ) -> Result<usize, RemoveActionError<A>> {
        // Determine how late the action can go: just past the last consecutive action it
        // commutes with. The centralizer borrows the action, so it must be dropped before the
        // reordering below.
        let commuting = {
            let centralizer = A::Centralizer::for_action(&self.actions[index]);
            self.actions[index + 1..]
                .iter()
                .take_while(|other| centralizer.commutes(other))
                .count()
        };
        let target = index + commuting;
        if target == index {
            return Ok(index);
        }

        // Compute the state at version `index`, i.e. the state the action was applied to.
        let state_at_index = match self.try_get_state_with(Version(index), context) {
            Ok(state) => state,
            // Unreachable: `index` is in range, so `Version(index)` is too.
            Err(GetStateError::VersionOutOfRange(_)) => {
                return Err(RemoveActionError::IndexOutOfRange(index));
            }
            Err(GetStateError::ActionFailed(error)) => {
                return Err(RemoveActionError::ActionFailed { index, error });
            }
        };

        // Shift the action toward `target`, one cached state at a time. Moving the action from
        // `position` to a cached version `v` only invalidates the cached state at `v`: the old
        // state there has the shifted action applied somewhere in its middle, `Action::inverse`
        // shifts it out past the commuting actions (yielding the state at `v - 1` under the new
        // order), and re-applying `actions[v]` (the action that moves to position `v - 1`) yields
        // the state at `v`. Each iteration leaves the history fully consistent, so a failed
        // `apply` merely leaves the action reordered.
        let first_affected = self.states.partition_point(|(Version(v), _)| *v <= index);
        let mut position = index;
        for cache_index in first_affected..self.states.len() {
            let (Version(cached_version), state) = &self.states[cache_index];
            let cached_version = *cached_version;
            if cached_version > target {
                break;
            }
            let mut shifted = state.clone();
            self.actions[position].inverse(&state_at_index, &mut shifted);
            let shifted = match self.actions[cached_version].apply(shifted, context) {
                Ok(state) => state,
                Err(error) => {
                    return Err(RemoveActionError::ActionFailed {
                        index: position,
                        error,
                    });
                }
            };
            self.actions[position..=cached_version].rotate_left(1);
            self.states[cache_index].1 = shifted;
            position = cached_version;
        }

        // Finish the shift past any remaining commuting actions before the next cached state.
        self.actions[position..=target].rotate_left(1);
        Ok(target)
    }

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns an error if the index is out of range or applying an action fails while
    /// rebuilding the cached states; in that case, the action is not removed, but it may have
    /// been reordered past some of the actions it commutes with —
    /// [`RemoveActionError::ActionFailed`] carries its current index.
    #[doc = remove_example!(
        context_setup!(),
        "history.try_remove_action_with(0, &mut context).unwrap()",
        "Set(0, 1)",
        "Set(1, 2)"
    )]
    ///
    #[doc = remove_complexity!()]
    pub fn try_remove_action_with(
        &mut self,
        index: usize,
        context: &mut A::Context,
    ) -> Result<A, RemoveActionError<A>> {
        let Version(last_version) = self.last_version();
        if index >= last_version {
            return Err(RemoveActionError::IndexOutOfRange(index));
        }

        // Shift the action as late as it can go cheaply; only the actions after its new position
        // need to be replayed.
        let index = self.shift_late(index, context)?;

        // Cached states at or before `index` survive the removal untouched; the ones after it
        // describe a history that no longer exists and are rebuilt by replaying without the
        // removed action. Every fallible action application happens before any mutation, so on
        // error the action is still at `index`.
        let keep = self
            .states
            .partition_point(|(Version(version), _)| *version <= index);
        let plan = Self::plan_refill(
            &self.states[..keep],
            last_version - 1,
            |from, to, state, context| {
                // Version `v` of the shortened history is reached by the action at position `v`
                // while before `index` and by the one at `v + 1` from `index` on, so a replay
                // spans at most two runs of the actions as they stand now.
                let split = index.clamp(from, to);
                let state = A::apply_batch(&self.actions[from..split], state, context)?;
                let tail = from.max(index);
                if tail < to {
                    A::apply_batch(&self.actions[tail + 1..to + 1], state, context)
                } else {
                    Ok(state)
                }
            },
            context,
        )
        .map_err(|error| RemoveActionError::ActionFailed { index, error })?;

        // All mutation happens below and is infallible.
        let action = self.actions.remove(index);
        self.install_refill(keep, plan);
        Ok(action)
    }

    /// Returns the most recent version.
    pub fn last_version(&self) -> Version {
        Version(self.actions.len())
    }

    /// Returns the most recent state.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    pub fn last_state(&self) -> &A::State {
        let (version, state) = self
            .states
            .last()
            .expect("the initial state is always cached");
        debug_assert_eq!(*version, self.last_version());
        state
    }

    /// Returns the state at the specified version.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*k* + log *n*) time, where *k* is how far back `version` is.
    pub fn try_get_state_with(
        &self,
        version: Version,
        context: &mut A::Context,
    ) -> Result<A::State, GetStateError<A>> {
        if version > self.last_version() {
            return Err(GetStateError::VersionOutOfRange(version));
        }
        // The initial state is always cached, so the partition is never empty.
        let index = self.states.partition_point(|(v, _)| *v <= version) - 1;
        let (Version(cached), state) = &self.states[index];
        A::apply_batch(&self.actions[*cached..version.0], state.clone(), context)
            .map_err(GetStateError::ActionFailed)
    }

    /// Returns the versions of the cached states.
    #[cfg(test)]
    pub(crate) fn cached_versions(&self) -> Vec<Version> {
        self.states.iter().map(|&(version, _)| version).collect()
    }

    /// Panics unless the cache is in the shape every operation must leave it in: strictly
    /// ascending, holding the initial and most recent states, and one or two states per window.
    /// The upper bound is what keeps the cache *O*(log *n*); the lower bound is what keeps
    /// [`Self::try_get_state_with`] linear in the distance back to the requested version.
    #[cfg(test)]
    pub(crate) fn assert_cache_invariant(&self) {
        let last = self.last_version();
        assert_eq!(self.states.first().map(|(v, _)| *v), Some(Version(0)));
        assert_eq!(self.states.last().map(|(v, _)| *v), Some(last));
        assert!(self.states.windows(2).all(|pair| pair[0].0 < pair[1].0));
        for scale in 0..Self::window_count(last.0) {
            let occupancy = self
                .states
                .iter()
                .filter(|(version, _)| Self::window_of(last, *version) == Some(scale))
                .count();
            assert!(
                (1..=2).contains(&occupancy),
                "window {scale} holds {occupancy} states at {last:?}: {:?}",
                self.cached_versions()
            );
        }
    }
}

impl<A: Action<Error = std::convert::Infallible>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    #[doc = push_example!(context_setup!(), "history.push_action_with(Add(42), &mut context)")]
    ///
    #[doc = push_complexity!()]
    pub fn push_action_with(&mut self, action: A, context: &mut A::Context) -> Version {
        // The state a push displaces from the end always stays cached — it is what lets the next
        // pop replay nothing — so, unlike a layout that evicts it, there is no clone to avoid
        // here and nothing to gain over the fallible implementation.
        self.try_push_action_with(action, context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!(
        context_setup!(),
        "history.pop_action_and_state_with(&mut context)",
        "(Add(42), 42)"
    )]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_action_and_state_with(&mut self, context: &mut A::Context) -> Option<(A, A::State)> {
        self.try_pop_action_and_state_with(context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!(
        context_setup!(),
        "history.pop_action_with(&mut context)",
        "Add(42)"
    )]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_action_with(&mut self, context: &mut A::Context) -> Option<A> {
        self.try_pop_action_with(context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!(context_setup!(), "history.pop_state_with(&mut context)", "42")]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_state_with(&mut self, context: &mut A::Context) -> Option<A::State> {
        self.try_pop_state_with(context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Removes the most recent `k` actions.
    ///
    /// Returns all actions that were removed, in reverse order, which may be fewer than `k` if
    /// there have been fewer than `k` actions.
    #[doc = pop_actions_example!(context_setup!(), "history.pop_actions_with(2, &mut context)")]
    ///
    #[doc = pop_actions_complexity!()]
    pub fn pop_actions_with(&mut self, k: usize, context: &mut A::Context) -> Vec<A> {
        self.try_pop_actions_with(k, context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range.
    #[doc = remove_example!(
        context_setup!(),
        "history.remove_action_with(0, &mut context)",
        "Some(Set(0, 1))",
        "Some(Set(1, 2))"
    )]
    ///
    #[doc = remove_complexity!()]
    pub fn remove_action_with(&mut self, index: usize, context: &mut A::Context) -> Option<A> {
        match self.try_remove_action_with(index, context) {
            Ok(action) => Some(action),
            Err(RemoveActionError::IndexOutOfRange(_)) => None,
            Err(RemoveActionError::ActionFailed { error, .. }) => match error {},
        }
    }

    /// Returns the state at the specified version, or `None` if there is no such version.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*k* + log *n*) time, where *k* is how far back `version` is.
    pub fn get_state_with(&self, version: Version, context: &mut A::Context) -> Option<A::State> {
        self.try_get_state_with(version, context).ok()
    }
}

impl<A: Action<Context = ()>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// Returns an error carrying the action back if applying it fails; in that case, the history
    /// is unchanged.
    #[doc = push_example!("", "history.try_push_action(Add(42)).unwrap()")]
    ///
    #[doc = push_complexity!()]
    pub fn try_push_action(&mut self, action: A) -> Result<Version, PushError<A>> {
        self.try_push_action_with(action, &mut ())
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!("", "history.try_pop_action_and_state().unwrap()", "(Add(42), 42)")]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_action_and_state(&mut self) -> Result<Option<(A, A::State)>, PopError<A>> {
        self.try_pop_action_and_state_with(&mut ())
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!("", "history.try_pop_action().unwrap()", "Add(42)")]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_action(&mut self) -> Result<Option<A>, PopError<A>> {
        self.try_pop_action_with(&mut ())
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
    #[doc = pop_example!("", "history.try_pop_state().unwrap()", "42")]
    ///
    #[doc = pop_complexity!()]
    pub fn try_pop_state(&mut self) -> Result<Option<A::State>, PopError<A>> {
        self.try_pop_state_with(&mut ())
    }

    /// Removes the most recent `k` actions.
    ///
    /// Returns all actions that were removed, in reverse order, which may be fewer than `k` if
    /// there have been fewer than `k` actions. On error, the history is unchanged.
    #[doc = pop_actions_example!("", "history.try_pop_actions(2).unwrap()")]
    ///
    #[doc = pop_actions_complexity!()]
    pub fn try_pop_actions(&mut self, k: usize) -> Result<Vec<A>, PopError<A>> {
        self.try_pop_actions_with(k, &mut ())
    }

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns an error if the index is out of range or applying an action fails while
    /// rebuilding the cached states; in that case, the action is not removed, but it may have
    /// been reordered past some of the actions it commutes with —
    /// [`RemoveActionError::ActionFailed`] carries its current index.
    #[doc = remove_example!(
        "",
        "history.try_remove_action(0).unwrap()",
        "Set(0, 1)",
        "Set(1, 2)"
    )]
    ///
    #[doc = remove_complexity!()]
    pub fn try_remove_action(&mut self, index: usize) -> Result<A, RemoveActionError<A>> {
        self.try_remove_action_with(index, &mut ())
    }

    /// Returns the state at the specified version.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*k* + log *n*) time, where *k* is how far back `version` is.
    pub fn try_get_state(&self, version: Version) -> Result<A::State, GetStateError<A>> {
        self.try_get_state_with(version, &mut ())
    }
}

impl<A: Action<Context = (), Error = std::convert::Infallible>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    #[doc = push_example!("", "history.push_action(Add(42))")]
    ///
    #[doc = push_complexity!()]
    pub fn push_action(&mut self, action: A) -> Version {
        self.push_action_with(action, &mut ())
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!("", "history.pop_action_and_state()", "(Add(42), 42)")]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_action_and_state(&mut self) -> Option<(A, A::State)> {
        self.pop_action_and_state_with(&mut ())
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!("", "history.pop_action()", "Add(42)")]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_action(&mut self) -> Option<A> {
        self.pop_action_with(&mut ())
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions.
    #[doc = pop_example!("", "history.pop_state()", "42")]
    ///
    #[doc = pop_complexity!()]
    pub fn pop_state(&mut self) -> Option<A::State> {
        self.pop_state_with(&mut ())
    }

    /// Removes the most recent `k` actions.
    ///
    /// Returns all actions that were removed, in reverse order, which may be fewer than `k` if
    /// there have been fewer than `k` actions.
    #[doc = pop_actions_example!("", "history.pop_actions(2)")]
    ///
    #[doc = pop_actions_complexity!()]
    pub fn pop_actions(&mut self, k: usize) -> Vec<A> {
        self.pop_actions_with(k, &mut ())
    }

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range.
    #[doc = remove_example!(
        "",
        "history.remove_action(0)",
        "Some(Set(0, 1))",
        "Some(Set(1, 2))"
    )]
    ///
    #[doc = remove_complexity!()]
    pub fn remove_action(&mut self, index: usize) -> Option<A> {
        self.remove_action_with(index, &mut ())
    }

    /// Returns the state at the specified version, or `None` if there is no such version.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*k* + log *n*) time, where *k* is how far back `version` is.
    pub fn get_state(&self, version: Version) -> Option<A::State> {
        self.get_state_with(version, &mut ())
    }
}

#[cfg(test)]
mod tests {
    // `proptest-derive`'s derived `Arbitrary` produces this warning.
    #![allow(non_local_definitions)]

    use super::*;
    use proptest::prelude::*;
    use proptest_derive::Arbitrary;

    #[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq)]
    struct TestAction(u8);

    impl Action for TestAction {
        type State = Vec<TestAction>;

        fn apply(
            &self,
            mut state: Self::State,
            _: &mut Self::Context,
        ) -> Result<Self::State, Self::Error> {
            state.push(*self);
            Ok(state)
        }
    }

    /// A [`TestAction`] that tallies its applications in the context, so that tests can assert on
    /// how much replaying an operation actually does.
    #[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq)]
    struct CountedAction(u8);

    impl Action for CountedAction {
        type State = Vec<CountedAction>;
        type Context = usize;

        fn apply(
            &self,
            mut state: Self::State,
            applications: &mut usize,
        ) -> Result<Self::State, Self::Error> {
            *applications += 1;
            state.push(*self);
            Ok(state)
        }
    }

    /// The failure of a `None` action, used to exercise the fallible paths.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct NoAction;

    impl std::fmt::Display for NoAction {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("there is no action to apply")
        }
    }

    impl std::error::Error for NoAction {}

    impl Action for Option<TestAction> {
        type State = Vec<TestAction>;
        type Error = NoAction;

        fn apply(
            &self,
            state: Self::State,
            context: &mut Self::Context,
        ) -> Result<Self::State, Self::Error> {
            let Some(action) = self else {
                return Err(NoAction);
            };
            action.apply(state, context).map_err(|e| match e {})
        }
    }

    #[derive(Arbitrary, Clone, Debug)]
    enum Step<A> {
        Push(A),
        Pop,
    }

    prop_compose! {
        fn history_strategy()(steps: Vec<Step<TestAction>>) -> History<TestAction> {
            let mut history = History::default();
            for step in steps {
                match step {
                    Step::Push(action) => {
                        history.push_action(action);
                    }
                    Step::Pop => {
                        history.pop_action();
                    }
                }
            }
            history
        }
    }

    impl Arbitrary for History<TestAction> {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            history_strategy().boxed()
        }
    }

    proptest! {
        #[test]
        fn push_action(mut history: History<TestAction>, action: TestAction) {
            let previous_version = history.last_version();
            let mut actions = Vec::from_iter(history.actions().cloned());
            let new_version = history.push_action(action);

            // This test only verifies the actions and last version are updated appropriately, not the
            // states. The consistency of the states with the actions is covered by other tests.
            prop_assert_eq!(history.last_version(), new_version);
            prop_assert!(new_version > previous_version);
            actions.push(action);
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }

        #[test]
        fn pop_action(mut history: History<TestAction>) {
            let previous_version = history.last_version();
            let mut actions = Vec::from_iter(history.actions().cloned());
            let action = history.pop_action();

            // This test only verifies the actions and last version are updated appropriately, not the
            // states. The consistency of the states with the actions is covered by other tests.
            prop_assert_eq!(action, actions.pop());
            prop_assert!(history.last_version() <= previous_version);
            prop_assert!(action.is_none() || history.last_version() < previous_version);
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }

        #[test]
        fn pop_actions(history: History<TestAction>, k in 0usize..48) {
            let mut singly = history.clone();
            let mut history = history;
            let popped = history.pop_actions_with(k, &mut ());

            let mut expected = Vec::new();
            for _ in 0..k {
                let Some(action) = singly.pop_action() else { break };
                expected.push(action);
            }

            // The two agree on everything observable. The cached versions need not match: the
            // layout depends on how the history got to its length, not only on the length.
            prop_assert_eq!(popped, expected);
            prop_assert_eq!(history.last_version(), singly.last_version());
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), singly.get_state(version));
            }
            history.assert_cache_invariant();
            singly.assert_cache_invariant();
        }

        /// Every operation must leave the cache in shape, whatever order they come in.
        #[test]
        fn cache_invariant_under_interleaving(steps: Vec<Step<TestAction>>) {
            let mut history = History::<TestAction>::default();
            history.assert_cache_invariant();
            for step in steps {
                match step {
                    Step::Push(action) => { history.push_action(action); }
                    Step::Pop => { history.pop_action(); }
                }
                history.assert_cache_invariant();
            }
        }

        /// Alternating a push with a pop must not rebuild anything: the state a push displaces
        /// from the end is exactly the one the pop after it needs back. A layout pinned to the
        /// length of the history fails this, replaying *O*(*n*) actions per cycle at the lengths
        /// that straddle a carry.
        #[test]
        fn alternating_push_and_pop_replays_nothing(n in 0usize..600, cycles in 1usize..8) {
            let mut applications = 0;
            let mut history = History::<CountedAction>::default();
            for i in 0..n {
                history.push_action_with(CountedAction((i % 256) as u8), &mut applications);
            }

            applications = 0;
            for _ in 0..cycles {
                history.push_action_with(CountedAction(0), &mut applications);
                history.pop_action_with(&mut applications);
                history.assert_cache_invariant();
                prop_assert_eq!(history.last_version(), Version(n));
            }

            // Only the applications the pushes themselves need; the pops replay nothing at all.
            prop_assert_eq!(applications, cycles);
        }

        /// Popping in bulk skips the cache states `k` successive pops compute and then discard, so
        /// it can only ever fall behind by the handful of windows it refills that they did not.
        #[test]
        fn pop_actions_costs_no_more_than_repeated_pops(n in 0usize..600, k in 0usize..600) {
            let mut applications = 0;
            let mut history = History::<CountedAction>::default();
            for i in 0..n {
                history.push_action_with(CountedAction((i % 256) as u8), &mut applications);
            }
            let mut singly = history.clone();

            applications = 0;
            history.pop_actions_with(k, &mut applications);
            let bulk = applications;

            applications = 0;
            for _ in 0..k.min(n) {
                singly.pop_action_with(&mut applications);
            }

            let allowance = History::<CountedAction>::window_count(n) as usize;
            prop_assert!(
                bulk <= applications + allowance,
                "bulk {bulk} exceeds {applications} one at a time by more than {allowance}",
            );
            history.assert_cache_invariant();
        }

        #[test]
        fn last_version(history: History<TestAction>) {
            prop_assert_eq!(Some(history.last_version()), history.versions().last());
        }

        #[test]
        fn get_state(history: History<TestAction>) {
            let mut state = Default::default();
            let mut version_iter = history.versions();
            let Some(initial_version) = version_iter.next() else {
                prop_assert!(false);
                return Ok(());
            };
            let actual_state = history.get_state(initial_version);
            prop_assert_eq!(actual_state.as_ref(), Some(&state));
            for (version, action) in version_iter.zip(history.actions()) {
                state = action.apply(state, &mut ()).unwrap();
                let actual_state = history.get_state(version);
                prop_assert_eq!(actual_state.as_ref(), Some(&state));
            }
        }
    }

    const SLOT_COUNT: usize = 4;

    /// An action which sets one of [`SLOT_COUNT`] slots; actions on distinct slots commute.
    #[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq)]
    struct SetSlot {
        slot: u8,
        value: u8,
    }

    impl SetSlot {
        fn slot(&self) -> usize {
            usize::from(self.slot) % SLOT_COUNT
        }
    }

    /// The centralizer of a [`SetSlot`]: every action on a different slot.
    struct OtherSlots(usize);

    impl<'a> Centralizer<'a, SetSlot> for OtherSlots {
        fn for_action(action: &'a SetSlot) -> Self {
            Self(action.slot())
        }

        fn commutes(&self, other: &SetSlot) -> bool {
            self.0 != other.slot()
        }
    }

    impl Action for SetSlot {
        type State = [u8; SLOT_COUNT];
        type Centralizer<'a> = OtherSlots;

        fn apply(
            &self,
            mut state: Self::State,
            _: &mut Self::Context,
        ) -> Result<Self::State, Self::Error> {
            state[self.slot()] = self.value;
            Ok(state)
        }

        fn inverse(&self, previous_state: &Self::State, state: &mut Self::State) {
            state[self.slot()] = previous_state[self.slot()];
        }
    }

    proptest! {
        #[test]
        fn try_remove_action(
            actions in prop::collection::vec(any::<SetSlot>(), 1..32),
            index in any::<prop::sample::Index>(),
        ) {
            let index = index.index(actions.len());
            let mut history = History::new([0; SLOT_COUNT]);
            for action in &actions {
                history.push_action(*action);
            }

            let mut expected_actions = actions.clone();
            let expected_removed = expected_actions.remove(index);
            prop_assert_eq!(
                history.try_remove_action_with(index, &mut ()).unwrap(),
                expected_removed
            );
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), expected_actions.clone());

            // The history must be indistinguishable from one built from the remaining actions,
            // apart from its cache layout, which depends on how it reached its current length.
            let mut expected = History::new([0; SLOT_COUNT]);
            for action in &expected_actions {
                expected.push_action(*action);
            }
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), expected.get_state(version));
            }
            history.assert_cache_invariant();
        }

        #[test]
        fn try_remove_action_out_of_range(mut history: History<TestAction>) {
            let Version(index) = history.last_version();
            let actions = Vec::from_iter(history.actions().cloned());
            prop_assert_eq!(history.try_remove_action_with(index, &mut ()), Err(RemoveActionError::IndexOutOfRange(index)));
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }
    }

    prop_compose! {
        fn fallible_history_strategy()(steps: Vec<Step<Option<TestAction>>>) -> History<Option<TestAction>> {
            let mut history = History::default();
            for step in steps {
                match step {
                    Step::Push(action) => {
                        let _ = history.try_push_action(action);
                    }
                    Step::Pop => {
                        let _ = history.try_pop_action();
                    }
                }
            }
            history
        }
    }

    impl Arbitrary for History<Option<TestAction>> {
        type Parameters = ();
        type Strategy = BoxedStrategy<Self>;
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            fallible_history_strategy().boxed()
        }
    }

    proptest! {
        #[test]
        fn try_push_action(mut history: History<Option<TestAction>>, action: Option<TestAction>) {
            let previous_version = history.last_version();
            let mut actions = Vec::from_iter(history.actions().cloned());
            let must_succeed = actions.iter().all(|a| a.is_some()) && action.is_some();
            if let Ok(new_version) = history.try_push_action(action) {
                prop_assert_eq!(history.last_version(), new_version);
                prop_assert!(new_version > previous_version);
                actions.push(action);
            } else {
                prop_assert!(!must_succeed);
            }
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }

        #[test]
        fn try_pop_action(mut history: History<Option<TestAction>>) {
            let previous_version = history.last_version();
            let mut actions = Vec::from_iter(history.actions().cloned());
            let must_succeed = actions.iter().all(|a| a.is_some());
            if let Ok(action) = history.try_pop_action() {
                prop_assert_eq!(action, actions.pop());
                prop_assert!(history.last_version() <= previous_version);
                prop_assert!(action.is_none() || history.last_version() < previous_version);
            } else {
                prop_assert!(!must_succeed);
            }
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }
    }

    /// A bulk pop rebuilds only the cache the shortened history ends up with, so unlike a run of
    /// single pops its cost does not grow with `k`.
    #[test]
    fn pop_actions_scales_better_than_repeated_pops() {
        for n in [64usize, 256, 1024, 4096] {
            let mut applications = 0;
            let mut history = History::<CountedAction>::default();
            for i in 0..n {
                history.push_action_with(CountedAction((i % 256) as u8), &mut applications);
            }
            let mut singly = history.clone();

            applications = 0;
            history.pop_actions_with(n / 2, &mut applications);
            let bulk = applications;

            applications = 0;
            for _ in 0..n / 2 {
                singly.pop_action_with(&mut applications);
            }

            // The margin grows with `n`; two is what the smallest size here already clears.
            assert!(
                bulk * 2 < applications,
                "at n = {n}, popping {} in bulk applied {bulk} actions against {applications} \
                 one at a time",
                n / 2,
            );
            history.assert_cache_invariant();
            for version in history.versions() {
                assert_eq!(
                    history.get_state_with(version, &mut 0),
                    singly.get_state_with(version, &mut 0),
                    "at n = {n}, {version:?}",
                );
            }
        }
    }

    /// Pins the behaviour at the lengths a layout pinned to the length of the history handles
    /// worst: those where `n + 1` is `3 * 2^(scale - 1)`, so that a push and the pop undoing it
    /// straddle a carry and the rigid layout rebuilds a state `2^(scale - 1)` actions back on
    /// every single cycle.
    #[test]
    fn alternation_is_cheap_at_pathological_lengths() {
        const CYCLES: usize = 64;
        for n in [10, 22, 46, 94, 190, 382, 766, 1534, 3070] {
            let mut applications = 0;
            let mut history = History::<CountedAction>::default();
            for i in 0..n {
                history.push_action_with(CountedAction((i % 256) as u8), &mut applications);
            }

            applications = 0;
            for _ in 0..CYCLES {
                history.push_action_with(CountedAction(0), &mut applications);
                history.pop_action_with(&mut applications);
            }

            history.assert_cache_invariant();
            assert_eq!(applications, CYCLES, "at n = {n}");
        }
    }

    /// Only `PushError` stores an action, so no other error type may bound the action type.
    /// Compiling at all is most of this test.
    #[test]
    fn errors_do_not_bound_the_action_type() {
        #[derive(Debug, PartialEq, Eq)]
        struct OpaqueError;

        impl std::fmt::Display for OpaqueError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("the action refused to apply")
            }
        }

        impl std::error::Error for OpaqueError {}

        /// Deliberately neither `Debug` nor `PartialEq`.
        struct Opaque(bool);

        impl Action for Opaque {
            type State = ();
            type Error = OpaqueError;

            fn apply(&self, _: (), _: &mut ()) -> Result<(), OpaqueError> {
                if self.0 { Ok(()) } else { Err(OpaqueError) }
            }
        }

        let mut history = History::<Opaque>::new(());
        assert!(history.try_push_action(Opaque(true)).is_ok());

        // `unwrap` on these needs only the *error* to be `Debug`, never the action.
        assert_eq!(history.try_get_state(Version(1)).unwrap(), ());
        assert_eq!(history.try_pop_action().unwrap().map(|a| a.0), Some(true));

        // A failed push hands the action back rather than dropping it.
        let error = history.try_push_action(Opaque(false)).err().unwrap();
        assert_eq!(*error.error(), OpaqueError);
        assert!(!error.into_action().0);
        assert_eq!(history.last_version(), Version(0));
        assert_eq!(history.last_version().index(), 0);
        assert_eq!(usize::from(history.last_version()), 0);
    }

    /// Every error that wraps a failed application must report it as its source, and say only
    /// which operation failed itself, so a chain-walking reporter prints each exactly once.
    #[test]
    fn failed_applications_are_reported_as_the_source() {
        use std::error::Error as _;

        fn chain(error: &dyn std::error::Error) -> Vec<String> {
            let mut chain = vec![error.to_string()];
            let mut source = error.source();
            while let Some(error) = source {
                chain.push(error.to_string());
                source = error.source();
            }
            chain
        }

        let mut history = History::<Option<TestAction>>::default();

        let error = history.try_push_action(None).unwrap_err();
        assert_eq!(
            chain(&error),
            ["failed to apply action", "there is no action to apply"]
        );
        assert!(error.source().unwrap().is::<NoAction>());

        // A pop only fails when a replay does, which needs an action that succeeds once and then
        // refuses, so the error is built directly rather than provoked.
        let error = PopError::<Option<TestAction>>(NoAction);
        assert_eq!(
            chain(&error),
            ["failed to apply action", "there is no action to apply"]
        );

        let error = RemoveActionError::<Option<TestAction>>::ActionFailed {
            index: 3,
            error: NoAction,
        };
        assert_eq!(
            chain(&error),
            ["failed to apply action", "there is no action to apply"]
        );

        let error = GetStateError::<Option<TestAction>>::ActionFailed(NoAction);
        assert_eq!(
            chain(&error),
            ["failed to apply action", "there is no action to apply"]
        );

        // The out-of-range errors have nothing underneath them.
        let error = History::<Option<TestAction>>::default()
            .try_get_state(Version(1))
            .unwrap_err();
        assert_eq!(chain(&error), ["version out of range"]);
        assert!(error.source().is_none());

        let error = History::<Option<TestAction>>::default()
            .try_remove_action(0)
            .unwrap_err();
        assert_eq!(chain(&error), ["action index out of range"]);
        assert!(error.source().is_none());
    }

    #[test]
    fn how_it_works() {
        let mut history = History::default();
        println!("{:?}", history.cached_versions());

        const N: usize = 15;

        for i in 1..=N {
            history.push_action(TestAction((i % 256) as u8));
            println!("{:?}", history.cached_versions());
        }

        for version in history.versions() {
            println!("{:?}", history.get_state(version));
        }

        for _ in 1..=N {
            history.pop_action();
            println!("{:?}", history.cached_versions());
        }
    }
}
