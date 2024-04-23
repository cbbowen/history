#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

use thiserror::Error;

/// An action which can be added to a [`History`].
pub trait Action {
    /// The type of state this action affects.
    type State: Clone;

    /// Applies this action to a state, producing a new state.
    fn apply(&self, state: Self::State) -> Self::State;
}

/// Identifies a specific version in the history.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(usize);

/// The only error type used by this library.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum Error {
    #[error("version out of range")]
    VersionOutOfRange(Version),
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
    /// #  fn apply(&self, state: i32) -> i32 { state }
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

    /// Returns all the versions from oldest to newest.
    pub fn versions(&self) -> impl Iterator<Item = Version> {
        (0..=self.actions.len()).map(Version)
    }

    /// Adds a new action to the end of the history and returns the new version.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  fn apply(&self, state: i32) -> i32 { self.0 + state }
    /// # }
    /// # let mut history = History::default();
    /// let before = *history.last_state();
    /// assert_eq!(*history.last_state(), 0);
    /// history.push_action(Add(42));
    /// assert_eq!(*history.last_state(), 42);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(1) amortized time.
    pub fn push_action(&mut self, action: A) -> Version {
        let new_state = action.apply(self.last_state().clone());
        self.actions.push(action);
        let new_version = Version(self.actions.len());

        // Restore invariants.
        let num_states = self.states.len();
        let index_to_remove =
            num_states as isize - (1 + (new_version.0 + 1).trailing_zeros()) as isize;
        debug_assert!(index_to_remove > 0 || index_to_remove == -1);
        if index_to_remove > 0 {
            // This takes `O(k)` time but `k` is exponentially-distributed, so amortized, it is
            // `O(1)`.
            self.states.remove(index_to_remove as usize);
        }
        self.states.push((new_version, new_state));
        new_version
    }

    /// Removes and returns the most recent action or `None` if there are no actions.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, PartialEq, Eq)]
    /// # struct Add(i32);
    /// # impl Action for Add {
    /// #  type State = i32;
    /// #  fn apply(&self, state: i32) -> i32 { self.0 + state }
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
        let removed_version = self.actions.len();
        let action = self.actions.pop()?;
        self.states.pop();

        let index_to_insert =
            self.states.len() as isize - (removed_version + 1).trailing_zeros() as isize;
        if index_to_insert > 0 {
            let index_to_insert = index_to_insert as usize;
            let version_to_insert =
                removed_version + 1 - (1 << (1 + self.states.len() - index_to_insert));
            let (recent_version, state) = self.get_cached_state(index_to_insert - 1);
            let state = self.actions[recent_version.0..version_to_insert]
                .iter()
                .fold(state.clone(), |s, a| a.apply(s));
            self.states
                .insert(index_to_insert, (Version(version_to_insert), state));
        }
        Some(action)
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
    fn get_recent_state_index(&self, version: Version) -> Result<usize, Error> {
        if version > self.last_version() {
            return Err(Error::VersionOutOfRange(version));
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
    pub fn get_state(&self, version: Version) -> Result<A::State, Error> {
        let state_index = self.get_recent_state_index(version)?;
        let (recent_version, state) = self.get_cached_state(state_index);
        Ok(self.actions[recent_version.0..version.0]
            .iter()
            .fold(state.clone(), |s, a| a.apply(s)))
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
        type State = Vec<u8>;

        fn apply(&self, mut state: Self::State) -> Self::State {
            state.push(self.0);
            state
        }
    }

    #[derive(Arbitrary, Clone, Debug)]
    enum Step {
        Push(TestAction),
        Pop,
    }

    prop_compose! {
        fn history_strategy()(steps: Vec<Step>) -> History<TestAction> {
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

             // This test only verifies the actions and last version are updated appropriately, not the states. The consistency of the states with the actions is covered by other tests.
             prop_assert_eq!(history.last_version(), new_version);
             prop_assert!(new_version > previous_version);
             actions.push(action);
             prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }

        fn pop_action(mut history: History<TestAction>) {
             let previous_version = history.last_version();
             let mut actions = Vec::from_iter(history.actions().cloned());
             let action = history.pop_action();

             // This test only verifies the actions and last version are updated appropriately, not the states. The consistency of the states with the actions is covered by other tests.
             prop_assert_eq!(action, actions.pop());
             prop_assert!(history.last_version() <= previous_version);
             prop_assert!(action.is_none() || history.last_version() < previous_version);
             prop_assert_eq!(Vec::from_iter(history.actions().cloned()), actions);
        }

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
             prop_assert_eq!(actual_state.as_ref(), Ok(&state));
             for (version, action) in version_iter.zip(history.actions()) {
                state = action.apply(state);
                let actual_state = history.get_state(version);
                prop_assert_eq!(actual_state.as_ref(), Ok(&state));
             }
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
