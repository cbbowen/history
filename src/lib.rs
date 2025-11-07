#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

/// An action which can be added to a [`History`].
pub trait Action {
    /// The type of state this action affects.
    type State: Clone;
    type Context;
    type Error;

    /// Applies this action to a state, producing a new state.
    fn apply(
        &self,
        state: Self::State,
        context: &mut Self::Context,
    ) -> Result<Self::State, Self::Error>;
}

/// Identifies a specific version in the history.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(usize);

/// The error type of [`History::try_push_action`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
#[error("failed to apply action")]
pub struct PushError<A: Action> {
    action: A,
    error: A::Error,
}

/// The error type of [`History::get_state`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum GetStateError<A: Action> {
    #[error("version out of range")]
    VersionOutOfRange(Version),

    #[error("failed to apply action")]
    ActionFailed(A::Error),
}

/// The error type of [`History::try_pop_action_and_state`], [`History::try_pop_action`], and
/// [`History::try_pop_state`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
#[error("failed to apply action")]
pub struct PopError<A: Action>(A::Error);

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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(state) }
    /// # }
    /// let history = History::<NoOp>::new(42);
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    pub fn new(initial: A::State) -> Self {
        Self {
            actions: Vec::new(),
            states: [(Version::default(), initial)].into_iter().collect(),
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

    // Returns the index in the cache that will be removed at the given version.
    fn cache_index_removed_at_version(version: usize) -> Option<usize> {
        if version == 0 {
            return None;
        }
        let cache_len = version.ilog2();
        let index = cache_len as i32 - (version + 1).trailing_zeros() as i32;
        if index == -1 {
            return None;
        }
        debug_assert!(index > 0);
        Some(index as usize)
    }

    /// Adds a new action to the end of the history and returns the new version.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// assert_eq!(*history.last_state(), 0);
    /// let mut context = ();
    /// let version = history.try_push_action_with(Add(42), &mut context).unwrap();
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
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

        // Determine which state will be removed.
        if let Some(index_to_remove) = Self::cache_index_removed_at_version(self.actions.len()) {
            // This takes `O(k)` time but `k` is exponentially-distributed, so amortized, it is
            // `O(1)`.
            self.states.remove(index_to_remove);
        }

        let new_version = Version(self.actions.len());
        self.states.push((new_version, new_state));
        Ok(new_version)
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// let mut context = ();
    /// assert_eq!(history.try_pop_action_and_state_with(&mut context).unwrap(), Some((Add(42), 42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_pop_action_and_state_with(
        &mut self,
        context: &mut A::Context,
    ) -> Result<Option<(A, A::State)>, PopError<A>> {
        let removed_version = self.actions.len();
        let new_version_and_state =
            Self::cache_index_removed_at_version(removed_version).map(|new_index| {
                let new_version = removed_version + 1 - (1 << (self.states.len() - new_index));
                let (Version(recent_version), state) = self.get_cached_state(new_index - 1);
                let state = self.actions[recent_version..new_version]
                    .iter()
                    .try_fold(state.clone(), |s, a| a.apply(s, context));
                state.map(|s| (new_index, (Version(new_version), s)))
            });
        let new_version_and_state = new_version_and_state.transpose().map_err(PopError)?;

        // Infallible because it does not perform any action applications. All mutation must
        // happen here.
        let infallible_portion = move || {
            let popped_action = self.actions.pop()?;
            let (_, popped_state) = self
                .states
                .pop()
                .expect("state/action size match invariant");

            if let Some((new_index, new_version_and_state)) = new_version_and_state {
                self.states.insert(new_index, new_version_and_state);
            }

            Some((popped_action, popped_state))
        };

        Ok(infallible_portion())
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions or an error if applying an action fails. In either
    /// case, the history is unchanged.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// let mut context = ();
    /// assert_eq!(history.try_pop_action_with(&mut context).unwrap(), Some(Add(42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// let mut context = ();
    /// assert_eq!(history.try_pop_action_with(&mut context).unwrap(), Some(Add(42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_pop_state_with(
        &mut self,
        context: &mut A::Context,
    ) -> Result<Option<A::State>, PopError<A>> {
        self.try_pop_action_and_state_with(context)
            .map(|o| o.map(|(_, s)| s))
    }

    /// Returns the most recent version.
    pub fn last_version(&self) -> Version {
        Version(self.actions.len())
    }

    fn get_cached_state(&self, state_index: usize) -> (Version, &A::State) {
        let (version, state) = self.states.get(state_index).unwrap();
        (*version, state)
    }

    /// Returns the most recent cached state not after the given `version` or an error if no such
    /// cached state exists.
    fn get_recent_state_index(&self, version: Version) -> Result<usize, GetStateError<A>> {
        if version > self.last_version() {
            return Err(GetStateError::VersionOutOfRange(version));
        }
        let index = self.states.partition_point(|(v, _)| *v <= version) - 1;
        Ok(index)
    }

    /// Returns the most recent state.
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) time.
    pub fn last_state(&self) -> &A::State {
        let (version, state) = self.states.last().unwrap();
        debug_assert_eq!(*version, self.last_version());
        state
    }

    /// Returns the state at the specified version.
    pub fn try_get_state_with(
        &self,
        version: Version,
        context: &mut A::Context,
    ) -> Result<A::State, GetStateError<A>> {
        let state_index = self.get_recent_state_index(version)?;
        let (recent_version, state) = self.get_cached_state(state_index);
        self.actions[recent_version.0..version.0]
            .iter()
            .try_fold(state.clone(), |s, a| a.apply(s, context))
            .map_err(GetStateError::ActionFailed)
    }

    /// Returns the versions of the cached states.
    #[cfg(test)]
    pub(crate) fn cached_versions(&self) -> Vec<Version> {
        self.states
            .iter()
            .map(|(version, _)| version)
            .cloned()
            .collect()
    }
}

impl<A: Action<Error = std::convert::Infallible>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// assert_eq!(*history.last_state(), 0);
    /// let mut context = ();
    /// let version = history.push_action_with(Add(42), &mut context);
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn push_action_with(&mut self, action: A, context: &mut A::Context) -> Version {
        self.try_push_action_with(action, context)
            .unwrap_or_else(|PushError { error, .. }| match error {})
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// let mut context = ();
    /// assert_eq!(history.pop_action_and_state_with(&mut context), Some((Add(42), 42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_action_and_state_with(&mut self, context: &mut A::Context) -> Option<(A, A::State)> {
        self.try_pop_action_and_state_with(context)
            .unwrap_or_else(|PopError(error)| match error {})
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// let mut context = ();
    /// assert_eq!(history.pop_action_with(&mut context), Some(Add(42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_action_with(&mut self, context: &mut A::Context) -> Option<A> {
        self.try_pop_action_with(context)
            .unwrap_or_else(|PopError(error)| match error {})
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// let mut context = ();
    /// history.push_action_with(Add(42), &mut context);
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_state(), Some(42));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_state_with(&mut self, context: &mut A::Context) -> Option<A::State> {
        self.try_pop_state_with(context)
            .unwrap_or_else(|PopError(error)| match error {})
    }

    /// Returns the state at the specified version.
    pub fn get_state_with(&self, version: Version, context: &mut A::Context) -> Option<A::State> {
        self.try_get_state_with(version, context).ok()
    }
}

impl<A: Action<Context = ()>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// assert_eq!(*history.last_state(), 0);
    /// let version = history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_push_action(&mut self, action: A) -> Result<Version, PushError<A>> {
        self.try_push_action_with(action, &mut ())
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_action_and_state(), Some((Add(42), 42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_pop_action_and_state(&mut self) -> Result<Option<(A, A::State)>, PopError<A>> {
        self.try_pop_action_and_state_with(&mut ())
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_action(), Some(Add(42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_pop_action(&mut self) -> Result<Option<A>, PopError<A>> {
        self.try_pop_action_with(&mut ())
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_state(), Some(42));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn try_pop_state(&mut self) -> Result<Option<A::State>, PopError<A>> {
        self.try_pop_state_with(&mut ())
    }

    /// Returns the state at the specified version.
    pub fn try_get_state(&self, version: Version) -> Result<A::State, GetStateError<A>> {
        self.try_get_state_with(version, &mut ())
    }
}

impl<A: Action<Context = (), Error = std::convert::Infallible>> History<A> {
    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  type Context = ();
    /// #  type Error = std::convert::Infallible;
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// assert_eq!(*history.last_state(), 0);
    /// let version = history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn push_action(&mut self, action: A) -> Version {
        self.push_action_with(action, &mut ())
    }

    /// Removes and returns the most recent action and the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_action_and_state(), Some((Add(42), 42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_action_and_state(&mut self) -> Option<(A, A::State)> {
        self.pop_action_and_state_with(&mut ())
    }

    /// Removes and returns the most recent action.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_action(), Some(Add(42)));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_action(&mut self) -> Option<A> {
        self.pop_action_with(&mut ())
    }

    /// Removes the most recent action and returns the state it produced.
    ///
    /// Returns `None` if there are no actions.
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
    /// #  fn apply(&self, state: i32, _: &mut ()) -> Result<i32, Self::Error> { Ok(self.0 + state) }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// assert_eq!(history.pop_state(), Some(42));
    /// assert_eq!(*history.last_state(), 0);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn pop_state(&mut self) -> Option<A::State> {
        self.pop_state_with(&mut ())
    }

    /// Returns the state at the specified version.
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
    use std::convert::Infallible;

    #[derive(Arbitrary, Clone, Copy, Debug, PartialEq, Eq)]
    struct TestAction(u8);

    impl Action for TestAction {
        type State = Vec<TestAction>;
        type Context = ();
        type Error = Infallible;

        fn apply(
            &self,
            mut state: Self::State,
            _: &mut Self::Context,
        ) -> Result<Self::State, Infallible> {
            state.push(*self);
            Ok(state)
        }
    }

    impl Action for Option<TestAction> {
        type State = Vec<TestAction>;
        type Context = ();
        type Error = ();

        fn apply(
            &self,
            state: Self::State,
            context: &mut Self::Context,
        ) -> Result<Self::State, Self::Error> {
            let Some(action) = self else { return Err(()) };
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
            let new_version = history.push_action(action.clone());

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
            if let Ok(new_version) = history.try_push_action(action.clone()) {
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
