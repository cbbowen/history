#![cfg_attr(
    feature = "associated_type_defaults",
    feature(associated_type_defaults)
)]
#![warn(missing_docs)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

macro_rules! context_doc {
    () => {
"Anything applying an action needs that the history does not hold, such as a resource pool or an interner. Every operation that can apply an action takes one by `&mut`.

Defaults to `()`, which unlocks the shorter methods that do not take one at all."
    }
}

macro_rules! error_doc {
    () => {
"The reason applying this action can fail.

Defaults to [`Infallible`](std::convert::Infallible), which unlocks the methods that return the result directly rather than a [`Result`].

This crate's error types report a value of this type as their [`source`](std::error::Error::source), so their own [`Display`](std::fmt::Display) says only which operation failed and leaves the reason to the chain."
    }
}

macro_rules! centralizer_doc {
    () => {
"Which other actions this one commutes with, and how to take this one back out of a state they have since been applied to. That is what lets the `remove_action` family correct the cached states after the action it removes rather than replay the actions that led to them.

Defaults to [`NonCommutative`], which never reports that anything commutes: always sound, and never cheaper."
    }
}

/// An action which can be added to a [`History`].
pub trait Action: Sized {
    /// The type of state this action affects.
    type State: Clone;

    #[cfg(feature = "associated_type_defaults")]
    #[doc = context_doc!()]
    type Context = ();
    #[cfg(not(feature = "associated_type_defaults"))]
    #[doc = context_doc!()]
    type Context;

    #[cfg(feature = "associated_type_defaults")]
    #[doc = error_doc!()]
    type Error: std::error::Error + 'static = std::convert::Infallible;
    #[cfg(not(feature = "associated_type_defaults"))]
    #[doc = error_doc!()]
    type Error: std::error::Error + 'static;

    #[cfg(feature = "associated_type_defaults")]
    #[doc = centralizer_doc!()]
    type Centralizer<'a>: Centralizer<'a, Self> = NonCommutative;
    #[cfg(not(feature = "associated_type_defaults"))]
    #[doc = centralizer_doc!()]
    type Centralizer<'a>: Centralizer<'a, Self>;

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
}

/// Identifies a specific version in the history.
///
/// Version `i` is the state reached by applying the first `i` actions. The action applied *to*
/// it, which leads to version `i + 1`, is the action with index `i`: that is the number
/// [`History::action`] and the `remove_action` family take. The action that *produced* version
/// `i` is therefore the one with index `i - 1`.
///
/// Versions count from the first action ever pushed, including the actions `forget_actions` has
/// since folded into the initial state, so a fold never renumbers anything. A version that names
/// a folded-away state stays meaningful — it is simply no longer reachable, and the operations
/// that take one say so rather than answering about a different state.
///
/// Shortening the history is another matter. After a pop, the next push reuses the number of the
/// version that went, and removing an action moves every later version down by one. A version
/// held across either names whichever state has its number now.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(usize);

impl Version {
    /// Returns the number of actions applied to reach this version, which is also the index of
    /// the action applied to it.
    pub fn index(self) -> usize {
        self.0
    }
}

impl From<Version> for usize {
    fn from(version: Version) -> Self {
        version.0
    }
}

/// Represents the set of actions an action commutes with, and takes that action back out of a
/// state they have since been applied to.
///
/// This is what lets the `remove_action` family correct the cached states that follow the action
/// it removes rather than replay the actions that led to them. The two halves cannot be
/// implemented apart, which is why they share a trait: [`Self::commutes`] promises something
/// about what [`Self::inverse`] does.
///
/// To be precise, let `c` be `Centralizer::for_action(a)` and let `b_1, ..., b_k` be any actions
/// with `c.commutes(b_i)` for every `i`. Write `bs(s)` for applying them in order to a state `s`.
/// Then for all `s`, taking `a` back out of the state reached by applying it and then them,
///
/// ```text
/// let mut state = bs(a.apply(s));
/// c.inverse(&s, &mut state);
/// ```
///
/// *must* leave `state` equivalent to `bs(s)`. Note that `inverse` is passed the state from
/// before `a` was applied, not the one immediately preceding `state`; the two differ by exactly
/// `b_1, ..., b_k`, which is what commuting has to make irrelevant. The single-action case
/// `k = 1` is not sufficient: the `remove_action` family takes an action out from behind a whole
/// run of commuting actions at once.
///
/// False negative results from `commutes` cost only time: the `remove_action` family replays the
/// actions it could have skipped. False positive results will produce incorrect states.
///
/// # Example
///
/// Actions that each set one element of a vector commute when they set different elements, and
/// taking one back out restores just the element it set.
///
/// ```
/// use history::*;
///
/// #[derive(Debug, PartialEq)]
/// struct Set(usize, i32);
///
/// struct OtherIndices(usize);
///
/// impl<'a> Centralizer<'a, Set> for OtherIndices {
///     fn for_action(action: &'a Set) -> Self {
///         Self(action.0)
///     }
///
///     fn commutes(&self, other: &Set) -> bool {
///         self.0 != other.0
///     }
///
///     fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
///         // Only this element. The others may have been set since, by actions that stay.
///         state[self.0] = previous_state[self.0];
///     }
/// }
///
/// impl Action for Set {
///     type State = Vec<i32>;
///     type Context = ();
///     type Error = std::convert::Infallible;
///     type Centralizer<'a> = OtherIndices;
///
///     fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
///         state[self.0] = self.1;
///         Ok(state)
///     }
/// }
///
/// let mut history = History::new(vec![0, 0]);
/// history.push_action(Set(0, 1));
/// history.push_action(Set(1, 2));
/// assert_eq!(*history.last_state(), vec![1, 2]);
///
/// // Taken out from behind `Set(1, 2)`, which is not applied again.
/// assert_eq!(history.remove_action(0), Some(Set(0, 1)));
/// assert_eq!(*history.last_state(), vec![0, 2]);
/// ```
pub trait Centralizer<'a, A: Action> {
    /// Returns the centralizer of `action`.
    fn for_action(action: &'a A) -> Self;

    /// Returns whether the action this centralizer was built for commutes with `other`.
    ///
    /// Returning `false` is always sound; see the trait documentation for what returning `true`
    /// promises.
    fn commutes(&self, other: &A) -> bool;

    /// Takes the action this centralizer was built for back out of `state`, in place.
    ///
    /// `previous_state` is the state the action was applied to, and `state` is what that became
    /// once the action and then one or more actions that [`Self::commutes`] accepts had been
    /// applied to it. Afterwards `state` must be equivalent to `previous_state` with only those
    /// other actions applied; the trait documentation states this precisely. Restoring from
    /// `previous_state` the part of the state the action affected is the usual way to do it.
    ///
    /// This is never called for a state that only the action itself has been applied to, so a
    /// centralizer whose `commutes` never returns `true` is never asked.
    fn inverse(&self, previous_state: &A::State, state: &mut A::State);
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

    // Never called, as nothing commutes, but restoring the whole state is what would be right:
    // with no other action to preserve, the state before the action is the state without it.
    fn inverse(&self, previous_state: &A::State, state: &mut A::State) {
        state.clone_from(previous_state);
    }
}

// These error types are written out rather than derived because every derive would bound the
// action type itself, which none of them stores except `PushError` and `PushActionsError`. Deriving
// `Debug` on `PopError<A>` would make `history.try_pop_action().unwrap()` require `A: Debug`.

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

/// The error type of [`History::try_push_actions`] and [`History::try_push_actions_with`].
///
/// A failed push adds none of the actions to the history, so the error carries them all back to
/// the caller, in the order they were given. It does not say which of them failed: they are
/// applied through [`Action::apply_batch`], which does not say either.
pub struct PushActionsError<A: Action> {
    actions: Vec<A>,
    error: A::Error,
}

impl<A: Action> PushActionsError<A> {
    /// Returns the actions that were not pushed.
    pub fn actions(&self) -> &[A] {
        &self.actions
    }

    /// Returns the error that applying one of the actions produced.
    pub fn error(&self) -> &A::Error {
        &self.error
    }

    /// Returns the actions that were not pushed, consuming the error.
    pub fn into_actions(self) -> Vec<A> {
        self.actions
    }

    /// Returns the error that applying one of the actions produced, consuming the rest.
    pub fn into_error(self) -> A::Error {
        self.error
    }

    /// Splits into the actions that were not pushed and the error one of them produced.
    pub fn into_parts(self) -> (Vec<A>, A::Error) {
        (self.actions, self.error)
    }
}

impl<A: Action> std::fmt::Display for PushActionsError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("failed to apply action")
    }
}

impl<A: Action + std::fmt::Debug> std::fmt::Debug for PushActionsError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PushActionsError")
            .field("actions", &self.actions)
            .field("error", &self.error)
            .finish()
    }
}

impl<A: Action + PartialEq> PartialEq for PushActionsError<A>
where
    A::Error: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.actions == other.actions && self.error == other.error
    }
}

impl<A: Action + Eq> Eq for PushActionsError<A> where A::Error: Eq {}

impl<A: Action + std::fmt::Debug> std::error::Error for PushActionsError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// The error type of [`History::try_get_state`] and [`History::try_get_state_with`].
pub enum GetStateError<A: Action> {
    /// The requested version is later than the most recent one.
    VersionOutOfRange(Version),

    /// The requested version is older than the initial one, [`History::forget_actions`] having
    /// folded it into the initial state.
    VersionForgotten(Version),

    /// Applying an action failed while reconstructing the state.
    ActionFailed(A::Error),
}

impl<A: Action> std::fmt::Display for GetStateError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VersionOutOfRange(_) => f.write_str("version out of range"),
            Self::VersionForgotten(_) => f.write_str("version already forgotten"),
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
            Self::VersionForgotten(version) => {
                f.debug_tuple("VersionForgotten").field(version).finish()
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
            (Self::VersionForgotten(a), Self::VersionForgotten(b)) => a == b,
            (Self::ActionFailed(a), Self::ActionFailed(b)) => a == b,
            _ => false,
        }
    }
}

impl<A: Action> Eq for GetStateError<A> where A::Error: Eq {}

impl<A: Action> std::error::Error for GetStateError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::VersionOutOfRange(_) | Self::VersionForgotten(_) => None,
            Self::ActionFailed(error) => Some(error),
        }
    }
}

/// The error type of [`History::try_remove_action`] and [`History::try_remove_action_with`].
pub enum RemoveActionError<A: Action> {
    /// No action has the requested index.
    IndexOutOfRange(usize),

    /// The action at the requested index is no longer in the history,
    /// [`History::forget_actions`] having folded it into the initial state.
    IndexForgotten(usize),

    /// Applying an action failed while rebuilding the cached states. The action to remove is
    /// still in the history, which is exactly as it was.
    ActionFailed(A::Error),
}

impl<A: Action> std::fmt::Display for RemoveActionError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfRange(_) => f.write_str("action index out of range"),
            Self::IndexForgotten(_) => f.write_str("action index already forgotten"),
            Self::ActionFailed(_) => f.write_str("failed to apply action"),
        }
    }
}

impl<A: Action> std::fmt::Debug for RemoveActionError<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfRange(index) => f.debug_tuple("IndexOutOfRange").field(index).finish(),
            Self::IndexForgotten(index) => f.debug_tuple("IndexForgotten").field(index).finish(),
            Self::ActionFailed(error) => f.debug_tuple("ActionFailed").field(error).finish(),
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
            (Self::IndexForgotten(a), Self::IndexForgotten(b)) => a == b,
            (Self::ActionFailed(a), Self::ActionFailed(b)) => a == b,
            _ => false,
        }
    }
}

impl<A: Action> Eq for RemoveActionError<A> where A::Error: Eq {}

impl<A: Action> std::error::Error for RemoveActionError<A> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IndexOutOfRange(_) | Self::IndexForgotten(_) => None,
            Self::ActionFailed(error) => Some(error),
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
#  type Context = ();
#  type Error = std::convert::Infallible;
#  type Centralizer<'a> = NonCommutative;
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

// The `# Example` section for the `push_actions` family. `call` must push `Add(40)` and then
// `Add(2)` and evaluate to the new version.
macro_rules! push_actions_example {
    ($setup:expr, $call:expr) => {
        concat!(
            "# Example

```
# use history::*;
# #[derive(Debug, PartialEq, Eq)]
# struct Add(i32);
# impl Action for Add {
#  type State = i32;
#  type Context = ();
#  type Error = std::convert::Infallible;
#  type Centralizer<'a> = NonCommutative;
#  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
# }
# let mut history = History::default();
",
            $setup,
            "let version = ",
            $call,
            ";
assert_eq!(*history.last_state(), 42);
assert_eq!(history.last_version(), version);
// Each action is still a version of its own.
assert_eq!(history.pop_action(), Some(Add(2)));
assert_eq!(*history.last_state(), 40);
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
#  type Context = ();
#  type Error = std::convert::Infallible;
#  type Centralizer<'a> = NonCommutative;
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
#  type Context = ();
#  type Error = std::convert::Infallible;
#  type Centralizer<'a> = NonCommutative;
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
#  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
#   state[self.0] = previous_state[self.0];
#  }
# }
# impl Action for Set {
#  type State = Vec<i32>;
#  type Centralizer<'a> = OtherIndices;
#  type Context = ();
#  type Error = std::convert::Infallible;
#  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
#   state[self.0] = self.1;
#   Ok(state)
#  }
# }
# let mut history = History::new(vec![0, 0]);
",
            $setup,
            "history.push_action(Set(0, 1));
history.push_action(Set(1, 2));
assert_eq!(*history.last_state(), vec![1, 2]);
// `Set(0, 1)` commutes with `Set(1, 2)`, so this removal applies nothing: see `Centralizer`.
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

macro_rules! push_actions_complexity {
    () => {
        "# Time complexity

Performs exactly one application for each action pushed and *O*(log *k*) state clones, where *k*
is how many there are, plus *O*(log *n*) bookkeeping. Pushing them one at a time clones the state
*k* times, to cache states that all but *O*(log *k*) of the later pushes then drop."
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

Applies no action more than once: the actions after `index` and, to reach them, the actions
between `index` and the cached state before it, of which there are fewer than three times as many.
That is *O*(*n*) applications, where *n* is the number of actions after `index`, plus the cache
repair a pop also performs, which is *O*(log *N*) amortized over the *N* actions the history holds
but can be a large part of the history for a single call.

The consecutive actions after `index` that the removed action commutes with are not replayed
where a cached state lies beyond them. Each cached state among them is corrected instead, with one
clone and one [`Centralizer::inverse`], and there are *O*(log *n*) of them. This saves a constant
factor rather than improving the bound: the corrections start from the state the removed action
was applied to, and reconstructing that is the replay from the cached state before `index`."
    };
}

// The hidden setup line the `_with` variants need, and the empty one the rest use.
macro_rules! context_setup {
    () => {
        "# let mut context = ();\n"
    };
}

const HAS_INITIAL_STATE: &str = "the initial state is always cached";

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
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  type Centralizer<'a> = NonCommutative;
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

    /// Returns the initial state, which is the state at [`Self::initial_version`].
    pub fn initial_state(&self) -> &A::State {
        &self.states.first().expect(HAS_INITIAL_STATE).1
    }

    /// Returns the oldest version the history still holds.
    ///
    /// This is [`Version::default`] until [`Self::forget_actions`] folds actions into the initial
    /// state, and the version of the state it folded them into thereafter. Every version before
    /// it is gone: the operations that take one report that rather than answering about a
    /// different state.
    pub fn initial_version(&self) -> Version {
        self.states.first().expect(HAS_INITIAL_STATE).0
    }

    // The number of actions the initial state already accounts for, and so the offset between a
    // version and the index of the action at that version in `self.actions`. Derived from the
    // cache rather than stored, so the two cannot disagree.
    fn base(&self) -> usize {
        self.initial_version().0
    }

    // The actions carrying the state at `from` to the state at `to`. Both must be versions this
    // history still holds. Every replay goes through here so that no version is used to index
    // `self.actions` directly.
    fn action_range(&self, from: usize, to: usize) -> &[A] {
        let base = self.base();
        &self.actions[from - base..to - base]
    }

    // Applies the actions carrying the state at `from` to the state at `to`, which must both be
    // versions this history still holds. Applies nothing, and so calls nothing, when they are the
    // same version.
    fn replay(
        &self,
        from: usize,
        to: usize,
        state: A::State,
        context: &mut A::Context,
    ) -> Result<A::State, A::Error> {
        if from == to {
            return Ok(state);
        }
        A::apply_batch(self.action_range(from, to), state, context)
    }

    /// Returns the action at `index`, which is the action that transforms the state at version
    /// `index` into the state at the following version. Returns `None` if the history does not
    /// hold it, either because there is no such action yet or because [`Self::forget_actions`]
    /// has folded it into the initial state.
    ///
    /// These are the indices the `remove_action` family takes. They count from the first action
    /// ever pushed, not from the oldest one the history still holds, so they are positions in
    /// [`Self::actions`] only until a fold: after one, the action at position `p` there has
    /// index `initial_version().index() + p`. Pairing [`Self::versions`] with [`Self::actions`]
    /// gives each action alongside the version whose index it has.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  type Centralizer<'a> = NonCommutative;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// history.push_action(Add(1));
    /// history.push_action(Add(2));
    /// history.push_action(Add(3));
    /// assert_eq!(history.action(0), Some(&Add(1)));
    ///
    /// // A fold drops the oldest actions without renumbering the rest.
    /// assert_eq!(history.forget_actions(1), vec![Add(1)]);
    /// assert_eq!(history.action(0), None);
    /// assert_eq!(history.action(1), Some(&Add(2)));
    ///
    /// // So to find the index of an action, look for it alongside the versions.
    /// let index = history
    ///     .versions()
    ///     .zip(history.actions())
    ///     .find(|(_, action)| **action == Add(3))
    ///     .map(|(version, _)| version.index());
    /// assert_eq!(index, Some(2));
    /// assert_eq!(history.remove_action(2), Some(Add(3)));
    /// ```
    pub fn action(&self, index: usize) -> Option<&A> {
        self.actions.get(index.checked_sub(self.base())?)
    }

    /// Returns all the actions the history still holds, from oldest to newest.
    ///
    /// [`Self::forget_actions`] drops the actions it folds into the initial state, so this does
    /// not necessarily start at [`Version::default`], and an action's position here is not
    /// necessarily its index: see [`Self::action`].
    pub fn actions(&self) -> impl ExactSizeIterator<Item = &A> {
        self.actions.iter()
    }

    /// Returns all the actions the history still holds, from oldest to newest.
    ///
    /// [`Self::forget_actions`] drops the actions it folds into the initial state, so this does
    /// not necessarily start at [`Version::default`].
    pub fn into_actions(self) -> impl ExactSizeIterator<Item = A> {
        self.actions.into_iter()
    }

    /// Returns all the versions from oldest to newest, starting at [`Self::initial_version`].
    pub fn versions(&self) -> impl ExactSizeIterator<Item = Version> {
        // Written this way to satisfy `ExactSizeIterator`.
        let base = self.base();
        (base..(base + self.actions.len() + 1)).map(Version)
    }

    // The cache holds states at geometrically spaced versions, dense near the most recent version
    // and sparse near the oldest. Rather than pinning each state to an exact version, it groups
    // them into *windows*: window `scale` covers the versions whose distance from the most recent
    // one lies in `2^scale..2^(scale + 1)`. No window ever holds more than two states, which is
    // what keeps the cache *O*(log *n*).
    //
    // Whatever shortens or reorders the history -- a pop, a bulk pop, a removal -- leaves every
    // window holding at least one state as well, rebuilding the ones that have run empty. A push
    // rebuilds nothing, so pushes can leave a window empty: the furthest state of a window moves
    // on into the next while the state behind it is still more than a window's width away. What
    // survives every operation is the weaker spacing that the bounds actually rest on: each
    // cached state is less than four times as far from the end as the next nearer one. Stated
    // exactly, cached states at consecutive distances `a < b` satisfy `b < 4 * (a + 1)`, so
    // reconstructing a state `k` back replays fewer than `3 * k` actions and folding down to the
    // most recent `m` actions leaves fewer than `4 * m`. It holds after a rebuild because every
    // window is then occupied, a push preserves it because it moves both states of a pair one
    // step further out, and dropping a state from a window that holds three preserves it because
    // the two it leaves share the window.
    //
    // Windows are measured backwards from the most recent version and counted off the number of
    // actions the history still holds, never off the version numbers themselves. That is what
    // lets `forget_actions` drop a prefix without disturbing the layout: the windows it does not
    // reach keep exactly the states they had.
    //
    // The redundancy is what keeps stepping back and forth cheap. A layout pinned to the length
    // of the history is a binary counter, so alternating a push with a pop re-propagates the same
    // carry forever, rebuilding a state *O*(*n*) actions back on every cycle. Here a push only
    // ever drops states a window can spare and a pop only rebuilds a window that has run empty,
    // and a rebuilt state is placed in the middle of its window, so it survives another
    // *O*(2^scale) operations in either direction before it must be rebuilt again.

    // The number of windows a history holding `len` actions has. This is the number the history
    // still holds, not the number of versions it has been through.
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

    // The versions at which a cache holding states at the versions `cached` lacks one that a
    // history running from `initial` to `new_len` needs, ascending: the midpoint of every window
    // that holds none of them, then `new_len` itself. `cached` must include `initial` and nothing
    // after `new_len`.
    //
    // A window is rebuilt at its midpoint so that the state survives another *O*(2^scale)
    // operations in either direction before the window runs empty again. Going from the coarsest
    // window to the finest is what makes the versions ascend, so that each state can be replayed
    // from the one before it. The coarsest window always holds the initial state, so it is never
    // rebuilt and the midpoint cannot underflow.
    fn refill_targets(
        cached: impl Iterator<Item = usize> + Clone,
        initial: usize,
        new_len: usize,
    ) -> impl Iterator<Item = usize> {
        // The most recent state must always be cached: `last_state` hands it out directly. It
        // belongs to no window, and it comes last because there is no later version.
        let last = (!cached.clone().any(|version| version == new_len)).then_some(new_len);
        (0..Self::window_count(new_len - initial))
            .rev()
            .filter(move |scale| {
                !cached
                    .clone()
                    .any(|version| (new_len - version) >> scale == 1)
            })
            .map(move |scale| new_len - ((3 << scale) >> 1))
            .chain(last)
    }

    // The newest of `states` and `plan` at or before `version`, which is the state a replay to
    // `version` has the least far to go from. There must be one.
    fn nearest_state<'s>(
        states: &'s [(Version, A::State)],
        plan: &'s [(Version, A::State)],
        version: usize,
    ) -> (usize, &'s A::State) {
        let (Version(from), state) = states
            .iter()
            .chain(plan)
            .filter(|(Version(cached), _)| *cached <= version)
            .max_by_key(|(cached, _)| *cached)
            .expect(HAS_INITIAL_STATE);
        (*from, state)
    }

    // Returns the states that must be added to `states` for it to cache a history whose most
    // recent version is `new_len`, ascending by version. `replay` must carry a state from one
    // version of the *new* history to a later one. The first of `states` is taken to be the
    // initial state, so passing only the most recent state plans the cache for the actions after
    // it alone.
    //
    // Every action application a cache repair needs happens here, so a caller can run this before
    // touching the history and leave it untouched if an application fails.
    fn plan_refill(
        states: &[(Version, A::State)],
        new_len: usize,
        mut replay: impl FnMut(usize, usize, A::State, &mut A::Context) -> Result<A::State, A::Error>,
        context: &mut A::Context,
    ) -> Result<Vec<(Version, A::State)>, A::Error> {
        // The oldest version the history holds, which the windows are counted off rather than off
        // `new_len`: the actions before it have been folded into the state cached there.
        let Version(initial) = states.first().expect(HAS_INITIAL_STATE).0;
        let cached = states.iter().map(|(Version(version), _)| *version);

        let mut plan: Vec<(Version, A::State)> = Vec::new();
        for target in Self::refill_targets(cached, initial, new_len) {
            // From the nearest state there is, the ones planned so far included. The targets
            // ascend, so no two replays overlap and no action is applied twice.
            let (from, state) = Self::nearest_state(states, &plan, target);
            let state = replay(from, target, state.clone(), context)?;
            plan.push((Version(target), state));
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
        // A plan entry can fall before a kept one: a window is refilled at its midpoint, and the
        // kept states nearer the end sit in finer windows. Nor does a removal's plan come in
        // order, since it holds the states it corrected alongside the ones it replayed. Unstable
        // is fine and avoids a scratch buffer -- versions are unique, so the order is
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
        let new_version = self.last_version();
        self.states.push((new_version, new_state));

        // A push never replays anything: the state that was most recent lands in the finest
        // window, and all that is left to do is drop the states a window can spare. That can
        // leave a coarser window empty, which the next operation that shortens the history
        // repairs; the note on the cache above says why no bound depends on it.
        self.prune_cache();
        Ok(new_version)
    }

    /// Adds new actions to the end of the history, in order, and returns the new most recent
    /// version. Each action is a version of its own, exactly as if they had been pushed one at a
    /// time.
    ///
    /// Returns an error carrying all of the actions back if applying one of them fails; in that
    /// case, the history is unchanged.
    #[doc = push_actions_example!(
        context_setup!(),
        "history.try_push_actions_with([Add(40), Add(2)], &mut context).unwrap()"
    )]
    ///
    #[doc = push_actions_complexity!()]
    pub fn try_push_actions_with(
        &mut self,
        actions: impl IntoIterator<Item = A>,
        context: &mut A::Context,
    ) -> Result<Version, PushActionsError<A>> {
        let actions = Vec::from_iter(actions);
        let Version(old_len) = self.last_version();
        let new_len = old_len + actions.len();

        // The new actions are a history of their own that starts at the most recent state, and
        // the states to cache for them are the ones that history would cache for itself. Seen
        // from the new end, the states already cached all lie beyond that one: they are still
        // the right way apart, since each has moved further out by the same amount, and all
        // they can need is pruning. Planning before touching anything is what leaves the history
        // unchanged if an application fails.
        let newest = &self.states[self.states.len() - 1..];
        let plan = Self::plan_refill(
            newest,
            new_len,
            |from, to, state, context| {
                A::apply_batch(&actions[from - old_len..to - old_len], state, context)
            },
            context,
        );
        match plan {
            Ok(plan) => {
                self.actions.extend(actions);
                self.states.extend(plan);
                self.prune_cache();
                Ok(Version(new_len))
            }
            Err(error) => Err(PushActionsError { actions, error }),
        }
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
        if self.actions.is_empty() {
            return Ok(None);
        }
        let new_len = self.last_version().0 - 1;
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
        // Saturates at the initial version rather than at zero: the actions before it are gone,
        // and popping cannot bring them back.
        let new_len = self.last_version().0 - k.min(self.actions.len());
        Ok(self.pop_to(new_len, context)?.actions)
    }

    // Shortens the history so that its most recent version is `new_len`, returning the actions
    // removed in reverse order along with the state the most recent one produced, or `None` if
    // nothing was removed. `new_len` must be a version the history holds.
    //
    // Every action application happens before any mutation, so on error the history is unchanged.
    fn pop_to(
        &mut self,
        new_len: usize,
        context: &mut A::Context,
    ) -> Result<Popped<A>, PopError<A>> {
        let base = self.base();
        debug_assert!(base <= new_len && new_len <= self.last_version().0);
        // Removing nothing must do nothing. Without this a call that pops no action would still
        // rebuild whichever windows pushes have left empty, applying actions -- and so possibly
        // failing -- on behalf of an operation that had nothing to do.
        if new_len == self.last_version().0 {
            return Ok(Popped {
                actions: Vec::new(),
                state: None,
            });
        }
        // Keep every cached state the shorter history can still use and rebuild only the windows
        // that running the end backwards leaves empty. When several actions go at once, that
        // skips the states the same number of successive pops would compute and then discard.
        let keep = self
            .states
            .partition_point(|(Version(version), _)| *version <= new_len);
        let plan = Self::plan_refill(
            &self.states[..keep],
            new_len,
            |from, to, state, context| self.replay(from, to, state, context),
            context,
        )
        .map_err(PopError)?;

        // All mutation happens below and is infallible. The most recent state is always cached
        // and something is coming off, so it is sitting at the end of the cache.
        debug_assert!(keep < self.states.len());
        let state = self.states.pop().map(|(version, state)| {
            debug_assert_eq!(version, self.last_version());
            state
        });
        let actions = self.actions.drain(new_len - base..).rev().collect();
        self.install_refill(keep, plan);
        Ok(Popped { actions, state })
    }

    /// Folds up to the `k` oldest actions into the initial state, dropping them, and returns the
    /// ones it folded in, oldest first.
    ///
    /// This is how a history that would otherwise grow without bound is kept to a size. The
    /// versions before [`Self::initial_version`] are gone afterwards: [`Self::try_get_state`]
    /// reports [`GetStateError::VersionForgotten`] for them, and the `remove_action` family
    /// reports [`RemoveActionError::IndexForgotten`]. The versions that remain keep the numbers
    /// they always had, so a [`Version`] held across a call still names the state it always
    /// named, or else says it is gone — it never comes to mean a different state.
    ///
    /// Only as many actions are folded in as can be without replaying any: the fold stops at the
    /// newest already-cached state at or before the `k`th action, so it may fold in fewer than
    /// `k`, or none at all when no state is cached in between. The shortfall is bounded, because
    /// the cache is geometrically spaced — asking to keep the most recent *m* actions leaves
    /// fewer than four times that many — so repeating the call as the history grows keeps it to a
    /// size. Ask for more than the history holds and it folds in everything, leaving the most
    /// recent state as the initial one.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  type Centralizer<'a> = NonCommutative;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// history.push_action(Add(1));
    /// history.push_action(Add(2));
    /// history.push_action(Add(39));
    /// let version = history.last_version();
    ///
    /// // The actions come back oldest first.
    /// assert_eq!(history.forget_actions(2), vec![Add(1), Add(2)]);
    /// assert_eq!(*history.initial_state(), 3);
    ///
    /// // Nothing observable about the versions that remain has changed.
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.last_version(), version);
    /// assert_eq!(history.get_state(history.initial_version()), Some(3));
    ///
    /// // The versions that were folded in are gone rather than renumbered.
    /// assert_eq!(history.get_state(Version::default()), None);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Performs no action applications and no state clones. Takes *O*(*m*) time, where *m* is the
    /// number of actions left afterwards, to shift them down, plus *O*(log *n*) bookkeeping.
    /// Folding away most of the history is therefore cheap and folding away a little is not, so
    /// fold in bulk.
    pub fn forget_actions(&mut self, k: usize) -> Vec<A> {
        let base = self.base();
        let target = base.saturating_add(k).min(self.last_version().0);
        // The newest cached state at or before `target`. The initial state is always cached, so
        // the partition is never empty, and its state is the one the folded actions produce.
        let index = self.states.partition_point(|(Version(v), _)| *v <= target) - 1;
        let Version(new_base) = self.states[index].0;

        self.states.drain(..index);
        let actions = self.actions.drain(..new_base - base).collect();

        // That is the whole repair. Windows are measured from the most recent version, which a
        // fold does not move, so every cached state that survives is exactly as far from the end
        // as it was and sits in the window it sat in. The states the fold dropped were the
        // furthest ones, so no surviving state has lost the nearer neighbour its spacing is
        // measured against, and no window has gained a state to prune.
        actions
    }

    /// Removes and returns the action at `index`, which is the action that transforms the state
    /// at version `index` into the state at the following version: the one [`Self::action`]
    /// returns for the same index. Later states are rebuilt as if the removed action had never
    /// been applied, and every later version moves down by one.
    ///
    /// Returns an error if the index is out of range, names an action a fold has already
    /// swallowed, or applying an action fails while rebuilding the cached states. In every case,
    /// the history is unchanged.
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
        // Actions are addressed by version throughout, so every index into `self.actions` is
        // offset by the actions the initial state has already absorbed.
        let base = self.base();
        if index < base {
            return Err(RemoveActionError::IndexForgotten(index));
        }
        let new_len = last_version - 1;

        // Cached states at or before `index` survive the removal untouched; the ones after it
        // describe a history that no longer exists. Everything that can fail happens while
        // planning their replacements, before any mutation, so on error the history is unchanged.
        let keep = self
            .states
            .partition_point(|(Version(version), _)| *version <= index);
        let plan = {
            let (kept, stale) = self.states.split_at(keep);

            // Version `v` of the shortened history is reached by the action at position `v`
            // while before `index` and by the one at `v + 1` from `index` on, so a replay spans
            // at most two runs of the actions as they stand now.
            let replay_without = |from: usize, to: usize, state: A::State, context: &mut _| {
                let state = self.replay(from, index.clamp(from, to), state, context)?;
                let tail = from.max(index);
                if tail < to {
                    self.replay(tail + 1, to + 1, state, context)
                } else {
                    Ok(state)
                }
            };

            // A stale state need not be thrown away if every action between the removed one and
            // it commutes with the removed one: the removed action can then be taken straight
            // back out of it, which leaves the state one version earlier of the shortened
            // history. The state directly after the removed action is the exception. Nothing
            // lies in between to commute with, and taking the action out of it gives back the
            // state the action was applied to, which every other inversion needs to begin with.
            let centralizer = A::Centralizer::for_action(&self.actions[index - base]);
            let commuting = self.actions[index + 1 - base..]
                .iter()
                .take_while(|other| centralizer.commutes(other))
                .count();
            let invertible = {
                let start = stale.partition_point(|(Version(version), _)| *version <= index + 1);
                let end = stale
                    .partition_point(|(Version(version), _)| *version <= index + 1 + commuting);
                &stale[start..end.max(start)]
            };

            // The versions the shortened history will have states for before any replay: the
            // kept ones, the inverted ones, and the state at `index` itself, which the inversions
            // need and which costs nothing more to keep.
            let cached = kept
                .iter()
                .map(|(Version(version), _)| *version)
                .chain((!invertible.is_empty()).then_some(index))
                .chain(invertible.iter().map(|(Version(version), _)| *version - 1));

            let mut plan: Vec<(Version, A::State)> = Vec::new();
            let mut inverted = invertible.is_empty();
            // The trailing `None` is there so that the inversions still happen when nothing
            // after `index` needs replaying at all.
            for target in Self::refill_targets(cached, base, new_len)
                .map(Some)
                .chain([None])
            {
                // Invert once the replays have reached `index`, and not before: the state at
                // `index` is then replayed from the nearest state planned on the way there
                // rather than from further back, and every later replay can start from an
                // inverted state. That is what keeps commuting from ever costing an application.
                if !inverted && target.is_none_or(|target| target > index) {
                    inverted = true;
                    let (from, state) = Self::nearest_state(kept, &plan, index);
                    let rebuilt = if from == index {
                        None
                    } else {
                        let state = replay_without(from, index, state.clone(), context)
                            .map_err(RemoveActionError::ActionFailed)?;
                        Some(state)
                    };
                    let previous_state = rebuilt.as_ref().unwrap_or(state);
                    let corrected: Vec<_> = invertible
                        .iter()
                        .map(|(Version(version), state)| {
                            let mut state = state.clone();
                            centralizer.inverse(previous_state, &mut state);
                            (Version(version - 1), state)
                        })
                        .collect();
                    plan.extend(rebuilt.map(|state| (Version(index), state)));
                    plan.extend(corrected);
                }
                let Some(target) = target else { break };
                let (from, state) = Self::nearest_state(kept, &plan, target);
                let state = replay_without(from, target, state.clone(), context)
                    .map_err(RemoveActionError::ActionFailed)?;
                plan.push((Version(target), state));
            }
            plan
        };

        // All mutation happens below and is infallible.
        let action = self.actions.remove(index - base);
        self.install_refill(keep, plan);
        Ok(action)
    }

    /// Returns the most recent version.
    pub fn last_version(&self) -> Version {
        Version(self.base() + self.actions.len())
    }

    /// Returns the most recent state.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    pub fn last_state(&self) -> &A::State {
        let (version, state) = self.states.last().expect(HAS_INITIAL_STATE);
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
        // Guarding this is not merely a courtesy: the partition below would be empty, and the
        // index one before it would underflow.
        if version < self.initial_version() {
            return Err(GetStateError::VersionForgotten(version));
        }
        // The initial state is always cached, so the partition is never empty.
        let index = self.states.partition_point(|(v, _)| *v <= version) - 1;
        let (Version(cached), state) = &self.states[index];
        self.replay(*cached, version.0, state.clone(), context)
            .map_err(GetStateError::ActionFailed)
    }

    /// Returns the versions of the cached states.
    #[cfg(test)]
    pub(crate) fn cached_versions(&self) -> Vec<Version> {
        self.states.iter().map(|&(version, _)| version).collect()
    }

    /// Panics unless the cache is in the shape every operation must leave it in: non-empty,
    /// strictly ascending, holding the most recent state and the one before it, at most two
    /// states per window, and each state less than four times as far from the end as the next
    /// nearer one (distances `a < b` of consecutive states satisfy `b < 4 * (a + 1)`). The bound
    /// on each window is what keeps the cache *O*(log *n*); the spacing is what keeps
    /// [`Self::try_get_state_with`] linear in the distance back to the requested version and
    /// bounds what [`Self::forget_actions`] leaves behind.
    ///
    /// That the *initial* state is cached needs no assertion: the oldest cached state is what
    /// defines it.
    #[cfg(test)]
    pub(crate) fn assert_cache_invariant(&self) {
        let last = self.last_version();
        let cached = self.cached_versions();
        assert!(!cached.is_empty());
        assert_eq!(cached.last(), Some(&last));
        assert!(cached.windows(2).all(|pair| pair[0] < pair[1]));
        if !self.actions.is_empty() {
            assert!(
                cached.contains(&Version(last.0 - 1)),
                "the state before {last:?} is not cached: {cached:?}"
            );
        }
        for pair in cached.windows(2) {
            let (far, near) = (last.0 - pair[0].0, last.0 - pair[1].0);
            assert!(
                far < 4 * (near + 1),
                "{:?} is too far behind {:?} at {last:?}: {cached:?}",
                pair[0],
                pair[1]
            );
        }
        for (scale, occupancy) in self.window_occupancy().into_iter().enumerate() {
            assert!(
                occupancy <= 2,
                "window {scale} holds {occupancy} states at {last:?}: {cached:?}"
            );
        }
    }

    /// Panics unless every window holds a state, on top of [`Self::assert_cache_invariant`].
    /// This is the shape every operation that rebuilds the cache leaves it in -- a pop, a bulk
    /// pop, a removal -- but not one a push preserves.
    #[cfg(test)]
    pub(crate) fn assert_every_window_occupied(&self) {
        self.assert_cache_invariant();
        for (scale, occupancy) in self.window_occupancy().into_iter().enumerate() {
            assert!(
                occupancy >= 1,
                "window {scale} is empty at {:?}: {:?}",
                self.last_version(),
                self.cached_versions()
            );
        }
    }

    /// Returns how many cached states each window holds, finest window first.
    #[cfg(test)]
    pub(crate) fn window_occupancy(&self) -> Vec<usize> {
        let last = self.last_version();
        let mut occupancy = vec![0; Self::window_count(self.actions.len()) as usize];
        for (version, _) in &self.states {
            if let Some(scale) = Self::window_of(last, *version) {
                occupancy[scale as usize] += 1;
            }
        }
        occupancy
    }
}

impl<A: Action<Error = std::convert::Infallible>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    #[doc = push_example!(context_setup!(), "history.push_action_with(Add(42), &mut context)")]
    ///
    #[doc = push_complexity!()]
    pub fn push_action_with(&mut self, action: A, context: &mut A::Context) -> Version {
        // The state a push displaces from the end always stays cached — it is the one the pop
        // that undoes the push hands back — so, unlike a layout that evicts it, there is no clone
        // to avoid here and nothing to gain over the fallible implementation.
        self.try_push_action_with(action, context)
            .unwrap_or_else(|error| match error.into_error() {})
    }

    /// Adds new actions to the end of the history, in order, and returns the new most recent
    /// version. Each action is a version of its own, exactly as if they had been pushed one at a
    /// time.
    #[doc = push_actions_example!(
        context_setup!(),
        "history.push_actions_with([Add(40), Add(2)], &mut context)"
    )]
    ///
    #[doc = push_actions_complexity!()]
    pub fn push_actions_with(
        &mut self,
        actions: impl IntoIterator<Item = A>,
        context: &mut A::Context,
    ) -> Version {
        self.try_push_actions_with(actions, context)
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

    /// Removes and returns the action at `index`, which is the action that transforms the state
    /// at version `index` into the state at the following version: the one [`Self::action`]
    /// returns for the same index. Later states are rebuilt as if the removed action had never
    /// been applied, and every later version moves down by one.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range or names an
    /// action [`Self::forget_actions`] has already folded into the initial state.
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
            Err(RemoveActionError::IndexOutOfRange(_) | RemoveActionError::IndexForgotten(_)) => {
                None
            }
            Err(RemoveActionError::ActionFailed(error)) => match error {},
        }
    }

    /// Returns the state at the specified version, or `None` if the history does not hold it —
    /// either because it is yet to come or because [`Self::forget_actions`] has folded it into
    /// the initial state.
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

    /// Adds new actions to the end of the history, in order, and returns the new most recent
    /// version. Each action is a version of its own, exactly as if they had been pushed one at a
    /// time.
    ///
    /// Returns an error carrying all of the actions back if applying one of them fails; in that
    /// case, the history is unchanged.
    #[doc = push_actions_example!(
        "",
        "history.try_push_actions([Add(40), Add(2)]).unwrap()"
    )]
    ///
    #[doc = push_actions_complexity!()]
    pub fn try_push_actions(
        &mut self,
        actions: impl IntoIterator<Item = A>,
    ) -> Result<Version, PushActionsError<A>> {
        self.try_push_actions_with(actions, &mut ())
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

    /// Removes and returns the action at `index`, which is the action that transforms the state
    /// at version `index` into the state at the following version: the one [`Self::action`]
    /// returns for the same index. Later states are rebuilt as if the removed action had never
    /// been applied, and every later version moves down by one.
    ///
    /// Returns an error if the index is out of range, names an action a fold has already
    /// swallowed, or applying an action fails while rebuilding the cached states. In every case,
    /// the history is unchanged.
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

    /// Adds new actions to the end of the history, in order, and returns the new most recent
    /// version. Each action is a version of its own, exactly as if they had been pushed one at a
    /// time.
    ///
    /// This is also what [`Extend`] does.
    #[doc = push_actions_example!("", "history.push_actions([Add(40), Add(2)])")]
    ///
    #[doc = push_actions_complexity!()]
    pub fn push_actions(&mut self, actions: impl IntoIterator<Item = A>) -> Version {
        self.push_actions_with(actions, &mut ())
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

    /// Removes and returns the action at `index`, which is the action that transforms the state
    /// at version `index` into the state at the following version: the one [`Self::action`]
    /// returns for the same index. Later states are rebuilt as if the removed action had never
    /// been applied, and every later version moves down by one.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range or names an
    /// action [`Self::forget_actions`] has already folded into the initial state.
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

    /// Returns the state at the specified version, or `None` if the history does not hold it —
    /// either because it is yet to come or because [`Self::forget_actions`] has folded it into
    /// the initial state.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*k* + log *n*) time, where *k* is how far back `version` is.
    pub fn get_state(&self, version: Version) -> Option<A::State> {
        self.get_state_with(version, &mut ())
    }
}

/// Pushes the actions as [`History::push_actions`] does.
impl<A: Action<Context = (), Error = std::convert::Infallible>> Extend<A> for History<A> {
    fn extend<I: IntoIterator<Item = A>>(&mut self, actions: I) {
        self.push_actions(actions);
    }
}

#[cfg(test)]
mod tests {
    // `proptest-derive`'s derived `Arbitrary` produces this warning.
    #![allow(non_local_definitions)]

    use super::*;

    use proptest::prelude::*;
    use proptest_derive::Arbitrary;
    use std::cell::Cell;
    use std::convert::Infallible;

    #[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq)]
    struct TestAction(u8);

    impl Action for TestAction {
        type State = Vec<TestAction>;

        // Defaults.
        type Context = ();
        type Error = Infallible;
        type Centralizer<'a> = NonCommutative;

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

        // Defaults.
        type Error = Infallible;
        type Centralizer<'a> = NonCommutative;

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

        // Defaults.
        type Context = ();
        type Centralizer<'a> = NonCommutative;

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

    /// One operation on a history. The ones that take a count or an index take it as a fraction
    /// of what the history holds at the time, so that they stay meaningful however long it has
    /// grown, and are weighted well below the others so that histories still reach interesting
    /// lengths.
    #[derive(Arbitrary, Clone, Debug)]
    enum Step<A> {
        #[proptest(weight = 12)]
        Push(A),
        #[proptest(weight = 2)]
        PushActions(Vec<A>),
        #[proptest(weight = 6)]
        Pop,
        #[proptest(weight = 1)]
        PopActions(prop::sample::Index),
        #[proptest(weight = 2)]
        Remove(prop::sample::Index),
        #[proptest(weight = 1)]
        Forget(prop::sample::Index),
    }

    impl<A: Action<Context = ()>> Step<A> {
        /// Performs this step, ignoring whether it succeeds: every operation leaves the history
        /// as it was when it fails.
        fn run(self, history: &mut History<A>) {
            let len = history.actions().len();
            match self {
                Step::Push(action) => {
                    let _ = history.try_push_action(action);
                }
                Step::PushActions(actions) => {
                    let _ = history.try_push_actions(actions);
                }
                Step::Pop => {
                    let _ = history.try_pop_action();
                }
                Step::PopActions(k) => {
                    let _ = history.try_pop_actions(k.index(len + 1));
                }
                Step::Remove(index) => {
                    if len > 0 {
                        let index = history.initial_version().index() + index.index(len);
                        let _ = history.try_remove_action(index);
                    }
                }
                Step::Forget(k) => {
                    history.forget_actions(k.index(len + 1));
                }
            }
        }
    }

    prop_compose! {
        /// A history that has been through an arbitrary run of operations. Some start from a
        /// run of pushes, because the steps alone rarely leave more than a handful of actions
        /// and the coarser windows of the cache only exist in a longer history.
        fn history_strategy()(
            prefill in prop::collection::vec(any::<TestAction>(), 0..150),
            steps: Vec<Step<TestAction>>,
        ) -> History<TestAction> {
            let mut history = History::default();
            for action in prefill {
                history.push_action(action);
            }
            for step in steps {
                step.run(&mut history);
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
            prop_assert_eq!(&popped, &expected);
            prop_assert_eq!(history.last_version(), singly.last_version());
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), singly.get_state(version));
            }
            if popped.is_empty() {
                history.assert_cache_invariant();
                singly.assert_cache_invariant();
            } else {
                history.assert_every_window_occupied();
                singly.assert_every_window_occupied();
            }
        }

        /// Every operation must leave the cache in shape, whatever order they come in, and the
        /// ones that rebuild it must leave no window empty.
        #[test]
        fn cache_invariant_under_interleaving(
            prefill in 0usize..300,
            steps: Vec<Step<TestAction>>,
        ) {
            let mut history = History::<TestAction>::default();
            history.assert_cache_invariant();
            for i in 0..prefill {
                history.push_action(TestAction((i % 256) as u8));
                history.assert_cache_invariant();
            }
            for step in steps {
                let before = history.actions().len();
                let rebuilds = matches!(step, Step::Pop | Step::PopActions(_) | Step::Remove(_));
                step.run(&mut history);
                if rebuilds && history.actions().len() < before {
                    history.assert_every_window_occupied();
                } else {
                    history.assert_cache_invariant();
                }
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
        /// it never applies more actions than they do. This is exact rather than approximate:
        /// with any allowance at all it also passes when the bulk pop replays a stretch twice.
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

            prop_assert!(
                bulk <= applications,
                "bulk {bulk} exceeds {applications} one at a time",
            );
            history.assert_cache_invariant();
        }

        #[test]
        fn last_version(history: History<TestAction>) {
            prop_assert_eq!(Some(history.last_version()), history.versions().last());
        }

        /// An action's index is the version it is applied to, wherever the history now starts,
        /// and it is the index that removes it.
        #[test]
        fn action(history: History<TestAction>, beyond in 0usize..4) {
            let Version(base) = history.initial_version();
            let Version(last) = history.last_version();
            prop_assert_eq!(history.versions().len(), history.actions().len() + 1);
            for (position, (version, action)) in
                history.versions().zip(history.actions()).enumerate()
            {
                prop_assert_eq!(version.index(), base + position);
                prop_assert_eq!(history.action(version.index()), Some(action));
                prop_assert_eq!(
                    history.clone().remove_action(version.index()),
                    Some(*action)
                );
            }
            prop_assert_eq!(history.action(last + beyond), None);
            prop_assert_eq!(history.action(usize::MAX), None);
            for index in base.saturating_sub(beyond + 1)..base {
                prop_assert_eq!(history.action(index), None);
            }
        }

        /// Folding must not disturb anything about the history it keeps. Every version that
        /// survives still names the state it always named, at the number it always had.
        #[test]
        fn forget_actions_preserves_the_versions_it_keeps(
            history: History<TestAction>,
            k in 0usize..48,
        ) {
            let before = history.clone();
            let mut history = history;
            let folded = history.forget_actions(k);

            // Never more than asked for, and exactly the prefix that is gone.
            prop_assert!(folded.len() <= k);
            prop_assert_eq!(
                folded.len(),
                history.initial_version().index() - before.initial_version().index()
            );
            prop_assert_eq!(
                Vec::from_iter(before.actions().take(folded.len()).cloned()),
                folded.clone()
            );
            prop_assert_eq!(
                Vec::from_iter(before.actions().skip(folded.len()).cloned()),
                Vec::from_iter(history.actions().cloned())
            );

            // The far end has not moved, and neither has any state still reachable.
            prop_assert_eq!(history.last_version(), before.last_version());
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), before.get_state(version));
            }
            prop_assert_eq!(
                Some(history.initial_state().clone()),
                before.get_state(history.initial_version())
            );

            // The versions that went are gone, not renumbered onto other states.
            for version in before.versions().take(folded.len()) {
                prop_assert_eq!(
                    history.try_get_state(version),
                    Err(GetStateError::VersionForgotten(version))
                );
                prop_assert_eq!(
                    history.clone().try_remove_action(version.index()),
                    Err(RemoveActionError::IndexForgotten(version.index()))
                );
            }
            history.assert_cache_invariant();
        }

        /// The shortfall from folding only as far as a cached state is bounded by the cache's
        /// geometric spacing, which is what makes repeating the call enough to keep a history to
        /// a size.
        #[test]
        fn forget_actions_leaves_a_bounded_remainder(n in 0usize..600, keep in 1usize..600) {
            let mut history = History::<TestAction>::default();
            for i in 0..n {
                history.push_action(TestAction((i % 256) as u8));
            }
            history.forget_actions(n.saturating_sub(keep));

            let remaining = history.actions().len();
            prop_assert!(remaining >= keep.min(n), "kept {remaining}, wanted {keep}");
            prop_assert!(remaining <= n, "kept {remaining} of {n}");
            prop_assert!(remaining < 4 * keep, "kept {remaining} to keep {keep}");
            history.assert_cache_invariant();
        }

        #[test]
        fn get_state(history: History<TestAction>) {
            // Not `Default::default()`: a fold leaves the actions it absorbed in the initial
            // state.
            let mut state = history.initial_state().clone();
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

    thread_local! {
        /// How many [`Slots`] this thread has cloned, and how many actions it has taken back
        /// out of one. A test that resets these before an operation sees only that operation.
        static CLONES: Cell<usize> = const { Cell::new(0) };
        static INVERSIONS: Cell<usize> = const { Cell::new(0) };
    }

    /// The state of a [`Mix`], which counts how often it is cloned.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Slots([u64; SLOT_COUNT]);

    impl Clone for Slots {
        fn clone(&self) -> Self {
            CLONES.set(CLONES.get() + 1);
            Self(self.0)
        }
    }

    /// The context of a [`Mix`]: how many applications there have been, and how many more
    /// there may be before they start to fail.
    #[derive(Debug)]
    struct Meter {
        applications: usize,
        budget: usize,
    }

    impl Meter {
        fn unlimited() -> Self {
            Self::limited(usize::MAX)
        }

        fn limited(budget: usize) -> Self {
            Self {
                applications: 0,
                budget,
            }
        }
    }

    /// The failure of a [`Mix`] applied on a spent [`Meter`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Exhausted;

    impl std::fmt::Display for Exhausted {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("the meter is spent")
        }
    }

    impl std::error::Error for Exhausted {}

    /// An action which folds a value into one of [`SLOT_COUNT`] slots. Folding, unlike
    /// overwriting, makes a slot depend on every action that has reached it and on their order,
    /// so an action that is applied twice, dropped, or moved past another on its slot shows in
    /// the state. Actions on distinct slots commute, which the centralizer reports only when
    /// `COMMUTES`: the two variants build identical histories and differ only in how much a
    /// removal is told.
    ///
    /// Whether an application succeeds is up to the [`Meter`], not the action, so one that
    /// succeeded when it was pushed can fail when it is replayed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Mix<const COMMUTES: bool> {
        slot: usize,
        value: u8,
    }

    impl<const COMMUTES: bool> Mix<COMMUTES> {
        fn strategy() -> impl Strategy<Value = Self> {
            (0..SLOT_COUNT, any::<u8>()).prop_map(|(slot, value)| Self { slot, value })
        }
    }

    /// The centralizer of a [`Mix`]: every action on a different slot, if it admits to any.
    struct OtherSlots<const COMMUTES: bool>(usize);

    impl<'a, const COMMUTES: bool> Centralizer<'a, Mix<COMMUTES>> for OtherSlots<COMMUTES> {
        fn for_action(action: &'a Mix<COMMUTES>) -> Self {
            Self(action.slot)
        }

        fn commutes(&self, other: &Mix<COMMUTES>) -> bool {
            COMMUTES && self.0 != other.slot
        }

        fn inverse(&self, previous_state: &Slots, state: &mut Slots) {
            INVERSIONS.set(INVERSIONS.get() + 1);
            state.0[self.0] = previous_state.0[self.0];
        }
    }

    impl<const COMMUTES: bool> Action for Mix<COMMUTES> {
        type State = Slots;
        type Context = Meter;
        type Error = Exhausted;
        type Centralizer<'a> = OtherSlots<COMMUTES>;

        fn apply(&self, mut state: Slots, meter: &mut Meter) -> Result<Slots, Exhausted> {
            meter.budget = meter.budget.checked_sub(1).ok_or(Exhausted)?;
            meter.applications += 1;
            let slot = &mut state.0[self.slot];
            *slot = slot
                .wrapping_mul(0x0100_0000_01b3)
                .wrapping_add(u64::from(self.value) + 1);
            Ok(state)
        }
    }

    /// Builds a history by pushing an action on each of `slots` in turn.
    fn mixed_history<const COMMUTES: bool>(
        slots: impl IntoIterator<Item = usize>,
    ) -> History<Mix<COMMUTES>> {
        let mut history = History::default();
        for (i, slot) in slots.into_iter().enumerate() {
            let value = (i % 251) as u8;
            history
                .try_push_action_with(Mix { slot, value }, &mut Meter::unlimited())
                .unwrap();
        }
        history
    }

    /// Panics unless every state `history` has cached -- and so every state it can hand out or
    /// replay from -- is the one its actions produce from its initial state.
    fn assert_cached_states_match_actions<const COMMUTES: bool>(history: &History<Mix<COMMUTES>>) {
        let base = history.initial_version().index();
        let mut state = history.initial_state().clone();
        let mut cached = history.states.iter().peekable();
        for offset in 0..=history.actions.len() {
            if offset > 0 {
                state = history.actions[offset - 1]
                    .apply(state, &mut Meter::unlimited())
                    .unwrap();
            }
            let version = Version(base + offset);
            if let Some((_, cached)) = cached.next_if(|(cached, _)| *cached == version) {
                assert_eq!(*cached, state, "the cached state at {version:?} is wrong");
            }
        }
        assert_eq!(cached.next().map(|(version, _)| version), None);
    }

    /// Some ways of spreading `n` actions over the slots, from nothing commuting to everything
    /// commuting with its neighbours.
    fn slot_patterns(n: usize) -> Vec<Vec<usize>> {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut scattered = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % SLOT_COUNT as u64) as usize
        };
        vec![
            vec![0; n],
            (0..n).map(|i| i % 2).collect(),
            (0..n).map(|i| i % SLOT_COUNT).collect(),
            (0..n).map(|i| (i / 5) % SLOT_COUNT).collect(),
            (0..n).map(|_| scattered()).collect(),
        ]
    }

    /// Some histories with the actions of `slots` in them, built the same way whatever
    /// `COMMUTES` is: by pushes alone, and with stretches popped and pushed again so that the
    /// cache is no longer laid out as pushes alone leave it.
    fn mixed_histories<const COMMUTES: bool>(slots: &[usize]) -> Vec<History<Mix<COMMUTES>>> {
        let pushed = mixed_history(slots.iter().copied());
        let n = slots.len();
        let meter = &mut Meter::unlimited();

        let mut bulk = pushed.clone();
        let popped = bulk.try_pop_actions_with(n / 3, meter).unwrap();
        for action in popped.into_iter().rev() {
            bulk.try_push_action_with(action, meter).unwrap();
        }

        let mut singly = pushed.clone();
        let mut popped = Vec::new();
        for _ in 0..n / 2 {
            popped.extend(singly.try_pop_action_with(meter).unwrap());
        }
        for action in popped.into_iter().rev() {
            singly.try_push_action_with(action, meter).unwrap();
        }

        vec![pushed, bulk, singly]
    }

    proptest! {
        #[test]
        fn try_remove_action(
            actions in prop::collection::vec(Mix::<true>::strategy(), 1..200),
            index in any::<prop::sample::Index>(),
        ) {
            let meter = &mut Meter::unlimited();
            let index = index.index(actions.len());
            let mut history = History::default();
            for action in &actions {
                history.try_push_action_with(*action, meter).unwrap();
            }

            let mut expected_actions = actions.clone();
            let expected_removed = expected_actions.remove(index);
            prop_assert_eq!(history.try_remove_action_with(index, meter), Ok(expected_removed));
            prop_assert_eq!(&history.actions, &expected_actions);

            // The history must be indistinguishable from one built from the remaining actions,
            // apart from its cache layout, which depends on how it reached its current length.
            let mut expected = History::default();
            for action in &expected_actions {
                expected.try_push_action_with(*action, meter).unwrap();
            }
            for version in history.versions() {
                prop_assert_eq!(
                    history.try_get_state_with(version, meter),
                    expected.try_get_state_with(version, meter)
                );
            }
            assert_cached_states_match_actions(&history);
            history.assert_every_window_occupied();
        }

        /// Removal addresses actions by version, so it must keep working once a fold has moved
        /// the actions out from under those numbers.
        #[test]
        fn try_remove_action_after_forgetting(
            actions in prop::collection::vec(Mix::<true>::strategy(), 1..200),
            forget in any::<prop::sample::Index>(),
            index in any::<prop::sample::Index>(),
        ) {
            let meter = &mut Meter::unlimited();
            let mut history = History::default();
            for action in &actions {
                history.try_push_action_with(*action, meter).unwrap();
            }
            // Never the whole history: there has to be an action left to remove, and folding
            // never takes more than it is asked for.
            let folded = history.forget_actions(forget.index(actions.len()));
            prop_assert!(history.actions().len() > 0);

            let initial = history.initial_state().clone();
            let base = history.initial_version().index();
            let index = base + index.index(history.actions().len());

            let mut expected_actions = history.actions.clone();
            let expected_removed = expected_actions.remove(index - base);
            prop_assert_eq!(history.try_remove_action_with(index, meter), Ok(expected_removed));
            prop_assert_eq!(&history.actions, &expected_actions);
            prop_assert_eq!(folded.len() + expected_actions.len() + 1, actions.len());

            // Indistinguishable from a history that started life at the folded state and never
            // saw the removed action, apart from where its versions are counted from.
            let mut expected = History::new(initial);
            for action in &expected_actions {
                expected.try_push_action_with(*action, meter).unwrap();
            }
            for (version, reference) in history.versions().zip(expected.versions()) {
                prop_assert_eq!(
                    history.try_get_state_with(version, meter),
                    expected.try_get_state_with(reference, meter)
                );
            }
            assert_cached_states_match_actions(&history);
            history.assert_every_window_occupied();
        }

        #[test]
        fn try_remove_action_out_of_range(
            mut history: History<TestAction>,
            beyond in 0usize..4,
        ) {
            let before = history.clone();
            for index in [history.last_version().index() + beyond, usize::MAX] {
                prop_assert_eq!(
                    history.try_remove_action(index),
                    Err(RemoveActionError::IndexOutOfRange(index))
                );
                prop_assert_eq!(history.remove_action(index), None);
            }
            prop_assert_eq!(&history.actions, &before.actions);
            prop_assert_eq!(&history.states, &before.states);
        }
    }

    /// Removing an action with nothing commuting must leave exactly the history that pushing the
    /// rest gives, from any index of any shape of history. This is the configuration every
    /// implementation that does not write a centralizer of its own has.
    #[test]
    fn removing_an_action_with_nothing_commuting() {
        for n in (1usize..=40).chain([64, 100]) {
            let slots: Vec<usize> = (0..n).map(|i| i % SLOT_COUNT).collect();
            for history in mixed_histories::<false>(&slots) {
                for index in 0..n {
                    let mut history = history.clone();
                    let mut expected = history.actions.clone();
                    let removed = expected.remove(index);

                    INVERSIONS.set(0);
                    assert_eq!(
                        history.try_remove_action_with(index, &mut Meter::unlimited()),
                        Ok(removed)
                    );
                    assert_eq!(INVERSIONS.get(), 0);
                    assert_eq!(history.actions, expected);
                    assert_eq!(history.last_version(), Version(n - 1));
                    assert_cached_states_match_actions(&history);
                    history.assert_every_window_occupied();
                }
            }
        }
    }

    /// Reporting that actions commute is only ever a saving. Whatever the centralizer admits to,
    /// a removal leaves the same states, and admitting to more never costs an application.
    #[test]
    fn commuting_never_costs_an_application() {
        for n in (1usize..=40).chain([64, 100, 257]) {
            for slots in slot_patterns(n) {
                let commuting = mixed_histories::<true>(&slots);
                let plain = mixed_histories::<false>(&slots);
                for (commuting, plain) in commuting.into_iter().zip(plain) {
                    for index in (0..n).step_by(n.div_ceil(64)) {
                        let mut commuting = commuting.clone();
                        let mut plain = plain.clone();
                        let mut saved = Meter::unlimited();
                        let mut spent = Meter::unlimited();

                        let removed = commuting
                            .try_remove_action_with(index, &mut saved)
                            .expect("the meter is unlimited");
                        let expected = plain
                            .try_remove_action_with(index, &mut spent)
                            .expect("the meter is unlimited");
                        assert_eq!(
                            (removed.slot, removed.value),
                            (expected.slot, expected.value)
                        );
                        assert_eq!(commuting.last_state(), plain.last_state());
                        assert_cached_states_match_actions(&commuting);
                        assert_cached_states_match_actions(&plain);
                        commuting.assert_every_window_occupied();
                        plain.assert_every_window_occupied();
                        assert!(
                            saved.applications <= spent.applications,
                            "removing {index} of {slots:?} applied {} actions where it applies \
                             {} with nothing commuting",
                            saved.applications,
                            spent.applications,
                        );
                    }
                }
            }
        }
    }

    /// An action that commutes with everything after it is taken back out of the cached states
    /// that follow it, so none of the actions after it is applied again. What a removal cannot
    /// avoid is rebuilding the state the action was applied to, and repairing the cache before
    /// it as a pop would: both replay only actions *before* the removed one, each at most once.
    #[test]
    fn removing_an_action_that_commutes_with_everything_after_it() {
        for n in (1usize..=64).chain([100, 257, 1000]) {
            let pushed = tallied_history::<true>(n);
            let mut tally = vec![0; n];
            for index in (0..n).step_by(n.div_ceil(64)) {
                let mut history = pushed.clone();
                tally.fill(0);
                assert_eq!(
                    history.remove_action_with(index, &mut tally),
                    Some(Tallied(index))
                );
                assert!(
                    tally[index..].iter().all(|&count| count == 0),
                    "removing {index} of {n} applied an action from it on: {tally:?}"
                );
                assert!(
                    tally[..index].iter().all(|&count| count <= 1),
                    "removing {index} of {n} applied an earlier action twice: {tally:?}"
                );
                history.assert_every_window_occupied();
            }

            // The corrections themselves are logarithmic: a clone and an inversion for each
            // cached state after the removed action, and there are at most two to a window.
            let lone = (0..n).map(|i| usize::from(i > 0));
            let mut history = mixed_history::<true>(lone);
            let cached = history.states.len();
            let mut meter = Meter::unlimited();
            CLONES.set(0);
            INVERSIONS.set(0);
            history
                .try_remove_action_with(0, &mut meter)
                .expect("the meter is unlimited");
            assert_eq!(meter.applications, 0, "removing the first of {n}");
            assert!(INVERSIONS.get() <= cached, "removing the first of {n}");
            assert!(CLONES.get() <= cached, "removing the first of {n}");
            assert!(
                cached <= 2 * History::<Mix<true>>::window_count(n) as usize + 1,
                "{cached} states cached for {n} actions"
            );
            assert_cached_states_match_actions(&history);
        }
    }

    /// A removal that fails must leave the history exactly as it was, wherever in its work the
    /// failure comes: before the state the action was applied to has been rebuilt, between two
    /// of the states it replays, or at the very last one.
    #[test]
    fn a_failed_removal_changes_nothing() {
        for n in 1usize..=24 {
            for slots in slot_patterns(n) {
                for history in mixed_histories::<true>(&slots) {
                    for index in 0..n {
                        let mut needed = Meter::unlimited();
                        let mut removed = history.clone();
                        removed
                            .try_remove_action_with(index, &mut needed)
                            .expect("the meter is unlimited");

                        for budget in 0..needed.applications {
                            let mut failed = history.clone();
                            assert_eq!(
                                failed.try_remove_action_with(index, &mut Meter::limited(budget)),
                                Err(RemoveActionError::ActionFailed(Exhausted))
                            );
                            assert_eq!(failed.actions, history.actions);
                            assert_eq!(failed.states, history.states);
                        }

                        let mut exact = history.clone();
                        let mut meter = Meter::limited(needed.applications);
                        assert!(exact.try_remove_action_with(index, &mut meter).is_ok());
                        assert_eq!(exact.states, removed.states);
                    }
                }
            }
        }
    }

    prop_compose! {
        fn fallible_history_strategy()(steps: Vec<Step<Option<TestAction>>>) -> History<Option<TestAction>> {
            let mut history = History::default();
            for step in steps {
                step.run(&mut history);
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
        /// A push fails exactly when its action does, and then hands the action back and
        /// leaves the history as it was.
        fn try_push_action(mut history: History<Option<TestAction>>, action: Option<TestAction>) {
            let before = history.clone();
            match history.try_push_action(action) {
                Ok(new_version) => {
                    prop_assert!(action.is_some());
                    prop_assert_eq!(history.last_version(), new_version);
                    prop_assert_eq!(new_version.index(), before.last_version().index() + 1);
                    prop_assert_eq!(history.actions().last(), Some(&action));
                    prop_assert_eq!(history.actions().len(), before.actions().len() + 1);
                }
                Err(error) => {
                    prop_assert_eq!(action, None);
                    prop_assert_eq!(*error.action(), None);
                    prop_assert_eq!(*error.error(), NoAction);
                    prop_assert_eq!(error.into_parts(), (None, NoAction));
                    prop_assert_eq!(&history.actions, &before.actions);
                    prop_assert_eq!(&history.states, &before.states);
                }
            }
        }

        /// An action that fails is never pushed, so a history of these never holds one and its
        /// pops cannot fail. What a pop does when a replay *does* fail is for the tests of
        /// [`Mix`], whose failures come from the context rather than from the action.
        #[test]
        fn try_pop_action(mut history: History<Option<TestAction>>) {
            prop_assert!(history.actions().all(Option::is_some));
            let previous_version = history.last_version();
            let mut actions = Vec::from_iter(history.actions().cloned());
            let action = history.try_pop_action().unwrap();
            prop_assert_eq!(action, actions.pop());
            prop_assert_eq!(
                history.last_version().index() + usize::from(action.is_some()),
                previous_version.index()
            );
            prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }
    }

    /// An action that tallies its applications by identity in the context, so that tests can
    /// tell *which* actions an operation replayed and how often, not merely how many. It has no
    /// state to get wrong, so it may claim to commute with everything, and does when `COMMUTES`.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Tallied<const COMMUTES: bool>(usize);

    /// The centralizer of a [`Tallied`]: every action or none.
    struct Indifferent<const COMMUTES: bool>;

    impl<'a, const COMMUTES: bool> Centralizer<'a, Tallied<COMMUTES>> for Indifferent<COMMUTES> {
        fn for_action(_: &'a Tallied<COMMUTES>) -> Self {
            Self
        }

        fn commutes(&self, _: &Tallied<COMMUTES>) -> bool {
            COMMUTES
        }

        fn inverse(&self, _: &(), _: &mut ()) {}
    }

    impl<const COMMUTES: bool> Action for Tallied<COMMUTES> {
        type State = ();
        type Context = Vec<u32>;
        type Error = Infallible;
        type Centralizer<'a> = Indifferent<COMMUTES>;

        fn apply(&self, _: (), tally: &mut Vec<u32>) -> Result<(), Infallible> {
            tally[self.0] += 1;
            Ok(())
        }
    }

    /// Builds a history by pushing `n` tallied actions, numbered in order.
    fn tallied_history<const COMMUTES: bool>(n: usize) -> History<Tallied<COMMUTES>> {
        let mut history = History::default();
        for i in 0..n {
            history.push_action_with(Tallied(i), &mut vec![0; n]);
        }
        history
    }

    /// No operation may apply an action twice to rebuild the cache: each state it needs is
    /// replayed from the nearest one it already has, including the ones it has only just
    /// planned. A removal with nothing commuting is the plainest case, since every action after
    /// the removed one has to be applied again exactly once.
    #[test]
    fn rebuilding_the_cache_applies_no_action_twice() {
        let sizes = (1usize..=64).chain([100, 257, 1000]);
        for n in sizes {
            let mut tally = vec![0; n];
            let pushed = tallied_history::<false>(n);
            let stride = n.div_ceil(64);

            for index in (0..n).step_by(stride) {
                let mut history = pushed.clone();
                tally.fill(0);
                assert_eq!(
                    history.remove_action_with(index, &mut tally),
                    Some(Tallied(index))
                );
                assert_eq!(
                    tally[index], 0,
                    "the removed action {index} of {n} was applied"
                );
                assert!(
                    tally[..index].iter().all(|&count| count <= 1),
                    "removing {index} of {n} applied an earlier action twice: {tally:?}"
                );
                assert!(
                    tally[index + 1..].iter().all(|&count| count == 1),
                    "removing {index} of {n} did not apply each later action once: {tally:?}"
                );
            }

            for k in (0..=n).step_by(stride) {
                let mut history = pushed.clone();
                tally.fill(0);
                history.pop_actions_with(k, &mut tally);
                assert!(
                    tally.iter().all(|&count| count <= 1),
                    "popping {k} of {n} applied an action twice: {tally:?}"
                );
            }
        }
    }

    proptest! {
        /// Pushing actions in bulk must leave exactly the history that pushing them one at a
        /// time does, apart from which states it has cached.
        #[test]
        fn push_actions(
            history: History<TestAction>,
            actions in prop::collection::vec(any::<TestAction>(), 0..200),
        ) {
            let mut singly = history.clone();
            let mut extended = history.clone();
            let mut history = history;

            let version = history.push_actions(actions.clone());
            extended.extend(actions.clone());
            for action in actions {
                singly.push_action(action);
            }

            prop_assert_eq!(version, singly.last_version());
            prop_assert_eq!(history.last_version(), singly.last_version());
            prop_assert_eq!(&history.actions, &singly.actions);
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), singly.get_state(version));
            }
            history.assert_cache_invariant();

            prop_assert_eq!(&extended.actions, &history.actions);
            prop_assert_eq!(&extended.states, &history.states);
        }
    }

    /// A bulk push applies each action it adds exactly once and nothing else, and clones only
    /// the states it will keep: one for each window of a history as long as the push. For a
    /// single action that is the one application and one clone of a plain push.
    #[test]
    fn push_actions_applies_each_action_once() {
        for n in [0usize, 1, 5, 64, 100] {
            for k in (0usize..=40).chain([64, 100, 1000]) {
                let mut tally = vec![0; n + k];
                let mut history = tallied_history::<false>(n);
                let version = history.push_actions_with((n..n + k).map(Tallied), &mut tally);
                assert_eq!(version, Version(n + k));
                assert!(
                    tally[..n].iter().all(|&count| count == 0),
                    "pushing {k} onto {n} applied an action already there: {tally:?}"
                );
                assert!(
                    tally[n..].iter().all(|&count| count == 1),
                    "pushing {k} onto {n} did not apply each action once: {tally:?}"
                );
                history.assert_cache_invariant();

                let mut history = mixed_history::<false>((0..n).map(|i| i % SLOT_COUNT));
                let actions = (0..k).map(|i| Mix {
                    slot: i % SLOT_COUNT,
                    value: (i % 251) as u8,
                });
                let mut meter = Meter::unlimited();
                CLONES.set(0);
                history
                    .try_push_actions_with(actions, &mut meter)
                    .expect("the meter is unlimited");
                assert_eq!(meter.applications, k);
                assert_eq!(
                    CLONES.get(),
                    History::<Mix<false>>::window_count(k) as usize,
                    "pushing {k} onto {n}"
                );
                assert_cached_states_match_actions(&history);
                history.assert_cache_invariant();
            }
        }
    }

    /// A bulk push that fails must add none of its actions, hand them all back in order, and
    /// leave the history exactly as it was, whichever of them it is that fails.
    #[test]
    fn a_failed_bulk_push_changes_nothing() {
        for n in [0usize, 1, 7, 33] {
            let history = mixed_history::<true>((0..n).map(|i| i % SLOT_COUNT));
            for k in 1usize..=40 {
                let actions: Vec<_> = (0..k)
                    .map(|i| Mix {
                        slot: i % SLOT_COUNT,
                        value: (i % 251) as u8,
                    })
                    .collect();
                for budget in 0..k {
                    let mut failed = history.clone();
                    let error = failed
                        .try_push_actions_with(actions.clone(), &mut Meter::limited(budget))
                        .expect_err("the meter runs out before the last action");
                    assert_eq!(error.actions(), actions);
                    assert_eq!(*error.error(), Exhausted);
                    assert_eq!(error.into_parts(), (actions.clone(), Exhausted));
                    assert_eq!(failed.actions, history.actions);
                    assert_eq!(failed.states, history.states);
                }
            }
        }
    }

    /// Builds a history by pushing `n` counted actions.
    fn counted_history(n: usize) -> History<CountedAction> {
        let mut history = History::default();
        for i in 0..n {
            history.push_action_with(CountedAction((i % 256) as u8), &mut 0);
        }
        history
    }

    /// Panics unless every version of `history` holds the state its actions produce.
    fn assert_states_match_actions(history: &History<CountedAction>) {
        let mut state = history.initial_state().clone();
        let mut versions = history.versions();
        assert_eq!(
            history.get_state_with(versions.next().unwrap(), &mut 0),
            Some(state.clone())
        );
        for (version, action) in versions.zip(history.actions()) {
            state.push(*action);
            assert_eq!(
                history.get_state_with(version, &mut 0).as_ref(),
                Some(&state)
            );
        }
        assert_eq!(*history.last_state(), state);
    }

    /// A push rebuilds nothing, so one that follows a bulk pop or a removal can carry the only
    /// state of a window out of it while the state behind is still too far back to follow. The
    /// cache must stay within its spacing all the same, every state must stay right, and the
    /// next operation that shortens the history must close the gap. These are the shortest
    /// sequences that open one.
    #[test]
    fn a_push_can_leave_a_window_empty() {
        type Sequence = fn(&mut History<CountedAction>);
        let sequences: [Sequence; 3] = [
            |history| {
                assert!(history.remove_action_with(0, &mut 0).is_some());
            },
            |history| {
                assert!(history.remove_action_with(0, &mut 0).is_some());
                assert_eq!(history.pop_actions_with(4, &mut 0).len(), 4);
            },
            |history| {
                assert_eq!(history.pop_actions_with(9, &mut 0).len(), 9);
                for _ in 0..8 {
                    history.push_action_with(CountedAction(0), &mut 0);
                }
                assert_eq!(history.pop_actions_with(6, &mut 0).len(), 6);
            },
        ];
        for (sequence, n) in sequences.into_iter().zip([16, 12, 14]) {
            let mut history = counted_history(n);
            sequence(&mut history);
            history.assert_every_window_occupied();

            history.push_action_with(CountedAction(1), &mut 0);
            assert!(
                history.window_occupancy().contains(&0),
                "the push after {n} left no window empty: {:?}",
                history.cached_versions()
            );
            history.assert_cache_invariant();
            assert_states_match_actions(&history);

            // Popping nothing is not an occasion to repair it: it must apply nothing and
            // change nothing.
            let cached = history.cached_versions();
            let mut applications = 0;
            assert!(history.pop_actions_with(0, &mut applications).is_empty());
            assert_eq!(applications, 0);
            assert_eq!(history.cached_versions(), cached);

            // Popping something is.
            assert!(history.pop_action_with(&mut 0).is_some());
            history.assert_every_window_occupied();
            assert_states_match_actions(&history);
        }
    }

    /// A bulk pop rebuilds only the cache the shortened history ends up with, so its cost grows
    /// more slowly with `k` than that of a run of single pops.
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

    /// Folding stops at a cached state, so it applies nothing at all — that is the whole reason
    /// it takes no context and cannot fail. It must also leave the cheap-alternation property
    /// intact, since it does not move the end of the history the windows are measured from.
    #[test]
    fn forget_actions_replays_nothing() {
        for n in [64usize, 256, 1024, 4096] {
            let mut applications = 0;
            let mut history = History::<CountedAction>::default();
            for i in 0..n {
                history.push_action_with(CountedAction((i % 256) as u8), &mut applications);
            }

            applications = 0;
            let folded = history.forget_actions(n / 2);
            assert_eq!(applications, 0, "at n = {n}");
            assert!(!folded.is_empty(), "at n = {n}, nothing was folded in");
            history.assert_cache_invariant();

            // The state the fold landed on is the one the actions it swallowed produce.
            let expected: Vec<_> = (0..folded.len())
                .map(|i| CountedAction((i % 256) as u8))
                .collect();
            assert_eq!(*history.initial_state(), expected, "at n = {n}");
            assert_eq!(
                history.initial_version().index(),
                folded.len(),
                "at n = {n}"
            );

            applications = 0;
            for _ in 0..64 {
                history.push_action_with(CountedAction(0), &mut applications);
                history.pop_action_with(&mut applications);
            }
            assert_eq!(applications, 64, "at n = {n}, after folding");
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

            // Defaults.
            type Context = ();
            type Centralizer<'a> = NonCommutative;

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

        // The other operations only fail when a replay does, which takes an action that
        // succeeds when it is pushed and refuses later.
        let spent = ["failed to apply action", "the meter is spent"];
        let pushed = mixed_history::<false>((0..64).map(|i| i % SLOT_COUNT));

        let error = pushed
            .clone()
            .try_push_actions_with(pushed.actions.clone(), &mut Meter::limited(1))
            .unwrap_err();
        assert_eq!(chain(&error), spent);
        assert!(error.source().unwrap().is::<Exhausted>());

        let mut history = pushed.clone();
        let error = loop {
            match history.try_pop_action_with(&mut Meter::limited(0)) {
                Ok(popped) => assert!(popped.is_some(), "no pop needed a replay"),
                Err(error) => break error,
            }
        };
        assert_eq!(chain(&error), spent);
        assert_eq!(*error.error(), Exhausted);
        assert_eq!(error.into_error(), Exhausted);

        let error = pushed
            .clone()
            .try_pop_actions_with(7, &mut Meter::limited(0))
            .unwrap_err();
        assert_eq!(chain(&error), spent);

        let error = pushed
            .clone()
            .try_remove_action_with(0, &mut Meter::limited(0))
            .unwrap_err();
        assert_eq!(error, RemoveActionError::ActionFailed(Exhausted));
        assert_eq!(chain(&error), spent);

        let error = pushed
            .versions()
            .find_map(|version| {
                pushed
                    .try_get_state_with(version, &mut Meter::limited(0))
                    .err()
            })
            .expect("no version needed a replay");
        assert_eq!(error, GetStateError::ActionFailed(Exhausted));
        assert_eq!(chain(&error), spent);

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

        // Nor do the ones a fold produces, which say that the state was reachable once rather
        // than that it never existed.
        let mut history = History::<Option<TestAction>>::default();
        history.try_push_action(Some(TestAction(1))).unwrap();
        history.try_push_action(Some(TestAction(2))).unwrap();
        assert_eq!(history.forget_actions(2).len(), 2);

        let error = history.try_get_state(Version(1)).unwrap_err();
        assert_eq!(chain(&error), ["version already forgotten"]);
        assert!(error.source().is_none());

        let error = history.try_remove_action(1).unwrap_err();
        assert_eq!(chain(&error), ["action index already forgotten"]);
        assert!(error.source().is_none());
    }

    /// A plain account of what a history of [`Mix`] actions should hold, to check one against.
    #[derive(Clone, Debug)]
    struct Model {
        base: usize,
        initial: Slots,
        actions: Vec<Mix<true>>,
    }

    impl Model {
        fn new() -> Self {
            Self {
                base: 0,
                initial: Slots::default(),
                actions: Vec::new(),
            }
        }

        fn last(&self) -> usize {
            self.base + self.actions.len()
        }

        /// The state at every version from the initial one on.
        fn states(&self) -> Vec<Slots> {
            let mut states = vec![self.initial.clone()];
            for action in &self.actions {
                let state = states.last().unwrap().clone();
                states.push(action.apply(state, &mut Meter::unlimited()).unwrap());
            }
            states
        }

        /// Panics unless `history` holds exactly this: the same actions from the same initial
        /// state at the same versions, and a cache within its invariant in which every state is
        /// the one those actions produce.
        fn assert_matches(&self, history: &History<Mix<true>>) {
            assert_eq!(history.initial_version(), Version(self.base));
            assert_eq!(history.last_version(), Version(self.last()));
            assert_eq!(history.actions, self.actions);
            let states = self.states();
            assert_eq!(*history.initial_state(), states[0]);
            assert_eq!(history.last_state(), states.last().unwrap());
            for (version, state) in &history.states {
                assert_eq!(
                    *state,
                    states[version.index() - self.base],
                    "the cached state at {version:?} is wrong"
                );
            }
            history.assert_cache_invariant();
        }
    }

    /// One operation on a history of [`Mix`] actions. As with [`Step`], counts and indices are
    /// fractions of what the history holds.
    #[derive(Clone, Debug)]
    enum Operation {
        Push(Mix<true>),
        PushActions(Vec<Mix<true>>),
        Pop,
        PopState,
        PopActions(prop::sample::Index),
        Remove(prop::sample::Index),
        Forget(prop::sample::Index),
        GetState(prop::sample::Index),
    }

    impl Operation {
        fn strategy() -> impl Strategy<Value = Self> {
            let index = any::<prop::sample::Index>;
            prop_oneof![
                12 => Mix::strategy().prop_map(Self::Push),
                2 => prop::collection::vec(Mix::strategy(), 0..48).prop_map(Self::PushActions),
                5 => Just(Self::Pop),
                1 => Just(Self::PopState),
                1 => index().prop_map(Self::PopActions),
                3 => index().prop_map(Self::Remove),
                1 => index().prop_map(Self::Forget),
                3 => index().prop_map(Self::GetState),
            ]
        }

        /// Performs this operation on both `history` and `model`, allowing it `budget`
        /// applications, and panics if the history does anything but what the model says it
        /// should: on success, the same results at a cost within what is documented, and on
        /// failure, nothing at all.
        fn run(self, history: &mut History<Mix<true>>, model: &mut Model, budget: usize) {
            let before = history.clone();
            let mut meter = Meter::limited(budget);
            let len = model.actions.len();
            let states = model.states();

            let outcome = match self {
                Self::Push(action) => history
                    .try_push_action_with(action, &mut meter)
                    .map(|version| {
                        assert_eq!(meter.applications, 1);
                        model.actions.push(action);
                        assert_eq!(version, Version(model.last()));
                    })
                    .map_err(|error| assert_eq!(error.into_parts(), (action, Exhausted))),
                Self::PushActions(actions) => history
                    .try_push_actions_with(actions.clone(), &mut meter)
                    .map(|version| {
                        assert_eq!(meter.applications, actions.len());
                        model.actions.extend(&actions);
                        assert_eq!(version, Version(model.last()));
                    })
                    .map_err(|error| assert_eq!(error.into_parts(), (actions, Exhausted))),
                Self::Pop => history
                    .try_pop_action_and_state_with(&mut meter)
                    .map(|popped| {
                        let state = states.last().unwrap().clone();
                        assert_eq!(popped, model.actions.pop().map(|action| (action, state)));
                        if len > 0 {
                            history.assert_every_window_occupied();
                        }
                    })
                    .map_err(|error| assert_eq!(error.into_error(), Exhausted)),
                Self::PopState => history
                    .try_pop_state_with(&mut meter)
                    .map(|popped| {
                        let state = states.last().unwrap().clone();
                        assert_eq!(popped, model.actions.pop().map(|_| state));
                    })
                    .map_err(|error| assert_eq!(error.into_error(), Exhausted)),
                Self::PopActions(k) => {
                    let k = k.index(len + 2);
                    history
                        .try_pop_actions_with(k, &mut meter)
                        .map(|popped| {
                            let kept = len.saturating_sub(k);
                            let mut expected = model.actions.split_off(kept);
                            expected.reverse();
                            assert_eq!(popped, expected);
                            if k == 0 {
                                assert_eq!(meter.applications, 0);
                                assert_eq!(history.states, before.states);
                            } else if len > 0 {
                                history.assert_every_window_occupied();
                            }
                        })
                        .map_err(|error| assert_eq!(error.into_error(), Exhausted))
                }
                Self::Remove(index) => {
                    // One past either end as well, which must be refused.
                    let index = (model.base + index.index(len + 2)).wrapping_sub(1);
                    let removed = history.try_remove_action_with(index, &mut meter);
                    if index >= model.last() {
                        assert_eq!(removed, Err(RemoveActionError::IndexOutOfRange(index)));
                        assert_eq!(meter.applications, 0);
                        Ok(())
                    } else if index < model.base {
                        assert_eq!(removed, Err(RemoveActionError::IndexForgotten(index)));
                        assert_eq!(meter.applications, 0);
                        Ok(())
                    } else {
                        removed
                            .map(|removed| {
                                assert_eq!(removed, model.actions.remove(index - model.base));
                                history.assert_every_window_occupied();
                            })
                            .map_err(|error| {
                                assert_eq!(error, RemoveActionError::ActionFailed(Exhausted))
                            })
                    }
                }
                Self::Forget(k) => {
                    let k = k.index(len + 2);
                    let folded = history.forget_actions(k);
                    assert!(folded.len() <= k);
                    assert_eq!(folded, model.actions[..folded.len()]);
                    // Asking to keep the most recent `m` leaves fewer than four times as many.
                    let (kept, wanted) = (len - folded.len(), len.saturating_sub(k));
                    assert!(kept >= wanted);
                    assert!(
                        kept == 0 || kept < 4 * wanted,
                        "{kept} actions left where {wanted} were to be kept"
                    );
                    model.actions.drain(..folded.len());
                    model.initial = states[folded.len()].clone();
                    model.base += folded.len();
                    Ok(())
                }
                Self::GetState(version) => {
                    let back = version.index(len + 1);
                    let version = Version(model.last() - back);
                    history
                        .try_get_state_with(version, &mut meter)
                        .map(|state| {
                            assert_eq!(state, states[len - back]);
                            // Linear in how far back the version is, whatever the cache holds.
                            assert!(
                                meter.applications < (3 * back).max(1),
                                "{} applications to go back {back}",
                                meter.applications
                            );
                        })
                        .map_err(|error| assert_eq!(error, GetStateError::ActionFailed(Exhausted)))
                }
            };

            match outcome {
                Ok(()) => assert!(meter.applications <= budget),
                Err(()) => {
                    // Only a spent meter makes anything fail, and a failure changes nothing.
                    assert_eq!(meter.applications, budget);
                    assert_eq!(history.actions, before.actions);
                    assert_eq!(history.states, before.states);
                }
            }
            model.assert_matches(history);
        }
    }

    proptest! {
        /// Whatever is done to a history, in whatever order and whether or not it succeeds, the
        /// history must go on holding exactly what a plain list of its actions says it should.
        /// About one operation in five is given too few applications to finish, so that every
        /// operation is also seen failing part of the way through its work.
        #[test]
        fn every_operation_agrees_with_a_model(
            prefill in prop::collection::vec(Mix::strategy(), 0..300),
            operations in prop::collection::vec(
                (Operation::strategy(), prop::option::weighted(0.2, 0usize..24)),
                0..120,
            ),
        ) {
            let mut history = History::default();
            let mut model = Model::new();
            Operation::PushActions(prefill).run(&mut history, &mut model, usize::MAX);
            for (operation, budget) in operations {
                operation.run(&mut history, &mut model, budget.unwrap_or(usize::MAX));
            }
        }
    }

    /// Every set of versions the cache of a history of `n` actions may hold: the initial version
    /// and the last two, at most two to a window, and each less than four times as far from the
    /// end as the next nearer one. Most of these no sequence of operations reaches, which is the
    /// point: the operations are meant to work on whatever the invariant admits, so that an
    /// argument from the invariant is an argument about them.
    fn admissible_caches(n: usize) -> Vec<Vec<usize>> {
        let interior = n.saturating_sub(2);
        (0u32..1 << interior)
            .map(|mask| {
                let mut versions = vec![0];
                versions.extend((1..=interior).filter(|version| mask >> (version - 1) & 1 == 1));
                versions.extend((n.saturating_sub(1)..=n).filter(|version| *version > 0));
                versions
            })
            .filter(|versions| {
                let spaced = versions
                    .windows(2)
                    .all(|pair| n - pair[0] < 4 * (n - pair[1] + 1));
                let crowded = (0..usize::BITS).any(|scale| {
                    let held = versions
                        .iter()
                        .filter(|version| (n - *version) >> scale == 1);
                    held.count() > 2
                });
                spaced && !crowded
            })
            .collect()
    }

    /// Builds a history of an action on each of `slots` whose cache holds exactly `versions`.
    fn history_with_cache<const COMMUTES: bool>(
        slots: &[usize],
        versions: &[usize],
    ) -> History<Mix<COMMUTES>> {
        let mut history = mixed_history::<COMMUTES>(slots.iter().copied());
        let meter = &mut Meter::unlimited();
        history.states = versions
            .iter()
            .map(|version| {
                let version = Version(*version);
                (version, history.try_get_state_with(version, meter).unwrap())
            })
            .collect();
        history.assert_cache_invariant();
        history
    }

    /// Every operation must do its job from any cache the invariant admits, and leave a cache
    /// the invariant admits behind. Checked for every such cache of every history short enough
    /// to enumerate them, against histories built the plain way.
    #[test]
    fn every_operation_works_from_every_admissible_cache() {
        let unlimited = Meter::unlimited;
        let mut caches = 0;
        for n in 0usize..=16 {
            let slots: Vec<usize> = (0..n).map(|i| (i * 7 / 3) % 3).collect();
            let plain = mixed_history::<true>(slots.iter().copied());
            let expected = |history: &History<Mix<true>>, version| {
                history.try_get_state_with(version, &mut unlimited())
            };

            for versions in admissible_caches(n) {
                caches += 1;
                let history = history_with_cache::<true>(&slots, &versions);
                let unhelped = history_with_cache::<false>(&slots, &versions);
                assert_cached_states_match_actions(&history);

                // Reading a state, in time linear in how far back it is.
                for version in history.versions() {
                    let mut meter = unlimited();
                    assert_eq!(
                        history.try_get_state_with(version, &mut meter),
                        expected(&plain, version)
                    );
                    let back = n - version.index();
                    assert!(
                        meter.applications < (3 * back).max(1),
                        "{} applications to go back {back} with {versions:?} cached",
                        meter.applications
                    );
                }

                // Pushing, one action and several.
                let pushed = Mix { slot: 1, value: 9 };
                let mut longer = history.clone();
                let mut meter = unlimited();
                longer.try_push_action_with(pushed, &mut meter).unwrap();
                assert_eq!(meter.applications, 1);
                assert_cached_states_match_actions(&longer);
                longer.assert_cache_invariant();
                for k in 0usize..=9 {
                    let mut longer = history.clone();
                    let mut meter = unlimited();
                    longer
                        .try_push_actions_with(vec![pushed; k], &mut meter)
                        .unwrap();
                    assert_eq!(meter.applications, k);
                    assert_eq!(longer.actions.len(), n + k);
                    assert_cached_states_match_actions(&longer);
                    longer.assert_cache_invariant();
                }

                // Popping, one action and several.
                for k in 0..=n + 1 {
                    let mut shorter = history.clone();
                    let popped = shorter.try_pop_actions_with(k, &mut unlimited()).unwrap();
                    assert_eq!(popped.len(), k.min(n));
                    assert_eq!(shorter.actions, history.actions[..n - k.min(n)]);
                    assert_cached_states_match_actions(&shorter);
                    if k == 0 {
                        assert_eq!(shorter.states, history.states);
                    } else {
                        shorter.assert_every_window_occupied();
                    }

                    if k == 1 {
                        let mut singly = history.clone();
                        let popped = singly.try_pop_action_and_state_with(&mut unlimited());
                        let state = history.last_state().clone();
                        assert_eq!(
                            popped,
                            Ok(history.actions.last().map(|last| (*last, state)))
                        );
                        assert_eq!(singly.states, shorter.states);
                    }
                }

                // Forgetting, which neither replays nor leaves more than it may.
                for k in 0..=n + 1 {
                    let mut folded = history.clone();
                    let forgotten = folded.forget_actions(k);
                    assert!(forgotten.len() <= k);
                    assert_eq!(forgotten, history.actions[..forgotten.len()]);
                    assert_eq!(folded.initial_version(), Version(forgotten.len()));
                    assert_eq!(folded.last_version(), Version(n));
                    assert_eq!(
                        Ok(folded.initial_state().clone()),
                        expected(&plain, folded.initial_version())
                    );
                    assert_cached_states_match_actions(&folded);
                    folded.assert_cache_invariant();
                    let (kept, wanted) = (folded.actions.len(), n.saturating_sub(k));
                    assert!(
                        kept == 0 || kept < 4 * wanted,
                        "{kept} left of {versions:?} where {wanted} were to be kept"
                    );
                }

                // Removing, with and without the help of a centralizer, and failing at every
                // point along the way.
                for index in 0..n {
                    let mut remaining = slots.clone();
                    remaining.remove(index);
                    let mut removed = history.clone();
                    let mut saved = unlimited();
                    assert_eq!(
                        removed.try_remove_action_with(index, &mut saved),
                        Ok(history.actions[index])
                    );
                    assert_cached_states_match_actions(&removed);
                    removed.assert_every_window_occupied();
                    assert_eq!(removed.actions.len(), n - 1);
                    assert_eq!(
                        removed
                            .actions
                            .iter()
                            .map(|action| action.slot)
                            .collect::<Vec<_>>(),
                        remaining
                    );

                    let mut unhelped = unhelped.clone();
                    let mut spent = unlimited();
                    unhelped.try_remove_action_with(index, &mut spent).unwrap();
                    assert_eq!(removed.last_state(), unhelped.last_state());
                    assert_cached_states_match_actions(&unhelped);
                    assert!(
                        saved.applications <= spent.applications,
                        "removing {index} with {versions:?} cached applied {} actions where it \
                         applies {} with nothing commuting",
                        saved.applications,
                        spent.applications,
                    );

                    for budget in 0..saved.applications {
                        let mut failed = history.clone();
                        assert_eq!(
                            failed.try_remove_action_with(index, &mut Meter::limited(budget)),
                            Err(RemoveActionError::ActionFailed(Exhausted))
                        );
                        assert_eq!(failed.states, history.states);
                        assert_eq!(failed.actions, history.actions);
                    }
                }
            }
        }
        // Enough of them that the enumeration is plainly not vacuous.
        assert!(caches > 3000, "only {caches} caches were admissible");
    }

    /// Undoing everything applies *O*(log *n*) actions per pop: no more than one for each window
    /// the history has, on average, and in fact about half that.
    #[test]
    fn popping_everything_is_logarithmic_per_pop() {
        for n in [1usize, 2, 3, 10, 100, 1000, 4096, 10_000] {
            let mut history = counted_history(n);
            let mut applications = 0;
            for remaining in (0..n).rev() {
                assert!(history.pop_action_with(&mut applications).is_some());
                assert_eq!(history.last_version(), Version(remaining));
            }
            assert_eq!(history.pop_action_with(&mut applications), None);
            let windows = History::<CountedAction>::window_count(n) as usize;
            assert!(
                applications <= n * windows,
                "popping all of {n} actions applied {applications}"
            );
        }
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
