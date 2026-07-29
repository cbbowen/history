#![feature(associated_type_defaults)]
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]

/// An action which can be added to a [`History`].
pub trait Action: Sized {
    /// The type of state this action affects.
    type State: Clone;
    type Context = ();
    type Error = std::convert::Infallible;

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

    /// Returns whether this action commutes with another.
    ///
    /// To be precise, for all `s`, `self.inverse(other.apply(self.apply(s)))` must be equivalent to
    /// `other.apply(s)`.
    fn commute(&self, other: &Self) -> bool {
        false
    }

    /// Applies the inverse of this action to a given state in place.
    ///
    /// For all `previous_state`, `self.inverse(previous_state, state)` where `state` is
    /// `self.apply(previous_state)` must leave `state` equivalent to `previous_state`. Observe that
    /// if `commute` always returns `false`, cloning `previous_state` into `state` is always a
    /// correct implementation. However, history surgery can be made more efficient with an
    /// implementation that only restores the portion of the state this action actually affects.
    fn inverse(&self, previous_state: &Self::State, state: &mut Self::State) {
        state.clone_from(previous_state);
    }
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

/// The error type of [`History::try_get_state`] and [`History::try_get_state_with`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum GetStateError<A: Action> {
    #[error("version out of range")]
    VersionOutOfRange(Version),

    #[error("failed to apply action")]
    ActionFailed(A::Error),
}

/// The error type of [`History::try_remove_action`] and [`History::try_remove_action_with`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum RemoveActionError<A: Action> {
    #[error("action index out of range")]
    IndexOutOfRange(usize),

    /// Applying an action failed while rebuilding the cached states. The action to remove is
    /// still in the history at `index`, which may differ from the requested index: the action
    /// may have been shifted past actions it commutes with.
    #[error("failed to apply action")]
    ActionFailed { index: usize, error: A::Error },
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
    fn cache_index_removed_at_version(Version(version): Version) -> Option<usize> {
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
        let new_version = Version(self.actions.len());
        if let Some(index_to_remove) = Self::cache_index_removed_at_version(new_version) {
            // This takes `O(k)` time but `k` is exponentially-distributed, so amortized, it is
            // `O(1)`.
            self.states.remove(index_to_remove);
        }

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
        let new_version_and_state = Self::cache_index_removed_at_version(Version(removed_version))
            .map(|new_index| {
                let new_version = removed_version + 1 - (1 << (self.states.len() - new_index));
                let (Version(recent_version), state) = self.get_cached_state(new_index - 1);
                let state = A::apply_batch(
                    &self.actions[recent_version..new_version],
                    state.clone(),
                    context,
                );
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
        // commutes with.
        let action = &self.actions[index];
        let commuting = self.actions[index + 1..]
            .iter()
            .take_while(|other| action.commute(other))
            .count();
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
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// # struct Set(usize, i32);
    /// # impl Action for Set {
    /// #  type State = Vec<i32>;
    /// #  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
    /// #   state[self.0] = self.1;
    /// #   Ok(state)
    /// #  }
    /// #  fn commute(&self, other: &Self) -> bool { self.0 != other.0 }
    /// #  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
    /// #   state[self.0] = previous_state[self.0];
    /// #  }
    /// # }
    /// let mut history = History::new(vec![0, 0]);
    /// history.push_action(Set(0, 1));
    /// history.push_action(Set(1, 2));
    /// assert_eq!(*history.last_state(), vec![1, 2]);
    /// let mut context = ();
    /// // `Set(0, 1)` commutes with `Set(1, 2)`, so this removal is cheap.
    /// assert_eq!(history.try_remove_action_with(0, &mut context).unwrap(), Set(0, 1));
    /// assert_eq!(*history.last_state(), vec![0, 2]);
    /// // Removing an action that does not commute replays the actions after it.
    /// history.push_action(Set(1, 4));
    /// assert_eq!(history.try_remove_action_with(0, &mut context).unwrap(), Set(1, 2));
    /// assert_eq!(*history.last_state(), vec![0, 4]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*n*) time, where *n* is the number of actions after `index`. The consecutive
    /// actions after `index` that the removed action commutes with are shifted past with only
    /// *O*(log *n*) state clones, inversions, and applications; the actions after the first
    /// non-commuting one are re-applied in full.
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

        // The cache transformation is the one a pop from `last_version` performs: the entry at
        // `last_version` is dropped and one previously evicted entry may be reinserted. In
        // addition, every cached state after `index` must be rebuilt without the removed action
        // by replaying the actions after `index`. Every fallible action application happens
        // before any mutation, so on error the action is still at `index`.
        let first_affected = self.states.partition_point(|(Version(v), _)| *v <= index);
        let last_cache_index = self.states.len() - 1;
        let reinserted = Self::cache_index_removed_at_version(Version(last_version)).map(
            |cache_index| {
                (
                    cache_index,
                    last_version + 1 - (1 << (self.states.len() - cache_index)),
                )
            },
        );

        let mut fixed_states = Vec::with_capacity(last_cache_index - first_affected);
        let mut reinserted_state = None;

        // A previously evicted entry at or before `index` is unaffected by the removal, so
        // reconstruct it from the old states, exactly as a pop would.
        if let Some((cache_index, reinserted_version)) = reinserted {
            if reinserted_version <= index {
                let (Version(recent_version), state) = &self.states[cache_index - 1];
                let state = A::apply_batch(
                    &self.actions[*recent_version..reinserted_version],
                    state.clone(),
                    context,
                );
                match state {
                    Ok(state) => reinserted_state = Some(state),
                    Err(error) => {
                        return Err(RemoveActionError::ActionFailed { index, error });
                    }
                }
            }
        }

        // Replay the actions after `index` over the state at `index`, recording the states the
        // new cache needs: one per cached version in `(index, last_version)`, plus the
        // reinserted version if it lies after `index`. In the new history, the state at such a
        // version `v` is the actions at old positions `index + 1..=v` applied to the state at
        // `index`.
        let replayed_reinserted_version = reinserted
            .map(|(_, version)| version)
            .filter(|version| *version > index);
        if first_affected < last_cache_index || replayed_reinserted_version.is_some() {
            // The reinserted version was evicted from the cache, so it never duplicates a cached
            // version.
            let mut replay_targets: Vec<(usize, bool)> = self.states
                [first_affected..last_cache_index]
                .iter()
                .map(|(Version(v), _)| (*v, false))
                .collect();
            if let Some(version) = replayed_reinserted_version {
                let position = replay_targets.partition_point(|(v, _)| *v < version);
                replay_targets.insert(position, (version, true));
            }

            let mut state = match self.try_get_state_with(Version(index), context) {
                Ok(state) => state,
                // Unreachable: `index` is in range, so `Version(index)` is too.
                Err(GetStateError::VersionOutOfRange(_)) => {
                    return Err(RemoveActionError::IndexOutOfRange(index));
                }
                Err(GetStateError::ActionFailed(error)) => {
                    return Err(RemoveActionError::ActionFailed { index, error });
                }
            };
            let mut version = index;
            for (target_version, is_reinserted) in replay_targets {
                state = match A::apply_batch(
                    &self.actions[version + 1..=target_version],
                    state,
                    context,
                ) {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(RemoveActionError::ActionFailed { index, error });
                    }
                };
                version = target_version;
                if is_reinserted {
                    reinserted_state = Some(state.clone());
                } else {
                    fixed_states.push(state.clone());
                }
            }
        }

        // All mutation happens below and is infallible.
        let action = self.actions.remove(index);
        for (offset, state) in fixed_states.into_iter().enumerate() {
            self.states[first_affected + offset].1 = state;
        }
        self.states
            .pop()
            .expect("state/action size match invariant");
        if let Some((cache_index, reinserted_version)) = reinserted {
            let state = reinserted_state.expect("reinserted state is computed above");
            self.states
                .insert(cache_index, (Version(reinserted_version), state));
        }
        Ok(action)
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

trait ResultExt<T> {
    fn unwrap_or_infallible(self) -> T;
}

impl<T> ResultExt<T> for Result<T, std::convert::Infallible> {
    fn unwrap_or_infallible(self) -> T {
        self.unwrap_or_else(|error| match error {})
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
        // The implementaiton below is morally equivalent to:
        // ```
        // self.try_push_action_with(action, context)
        //     .unwrap_or_else(|PushError { error, .. }| match error {});
        // ```
        // However, we can avoid a clone half the time.

        // Determine which state will be removed.
        self.actions.push(action);
        let new_version = Version(self.actions.len());
        let index_to_remove = Self::cache_index_removed_at_version(new_version);

        // If it's the last state, pop it instead of cloning then removing.
        let last_state = if index_to_remove.is_some_and(|i| i + 1 == self.states.len()) {
            self.states.pop().expect("non-empty").1
        } else {
            if let Some(index_to_remove) = index_to_remove {
                self.states.remove(index_to_remove);
            }
            self.states.last().expect("non-empty").1.clone()
        };

        let new_state = self
            .actions
            .last()
            .expect("non-empty")
            .apply(last_state, context)
            .unwrap_or_infallible();
        self.states.push((new_version, new_state));
        new_version
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

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// # struct Set(usize, i32);
    /// # impl Action for Set {
    /// #  type State = Vec<i32>;
    /// #  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
    /// #   state[self.0] = self.1;
    /// #   Ok(state)
    /// #  }
    /// #  fn commute(&self, other: &Self) -> bool { self.0 != other.0 }
    /// #  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
    /// #   state[self.0] = previous_state[self.0];
    /// #  }
    /// # }
    /// let mut history = History::new(vec![0, 0]);
    /// history.push_action(Set(0, 1));
    /// history.push_action(Set(1, 2));
    /// let mut context = ();
    /// assert_eq!(history.remove_action_with(0, &mut context), Some(Set(0, 1)));
    /// assert_eq!(*history.last_state(), vec![0, 2]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*n*) time, where *n* is the number of actions after `index`. The consecutive
    /// actions after `index` that the removed action commutes with are shifted past with only
    /// *O*(log *n*) state clones, inversions, and applications; the actions after the first
    /// non-commuting one are re-applied in full.
    pub fn remove_action_with(&mut self, index: usize, context: &mut A::Context) -> Option<A> {
        match self.try_remove_action_with(index, context) {
            Ok(action) => Some(action),
            Err(RemoveActionError::IndexOutOfRange(_)) => None,
            Err(RemoveActionError::ActionFailed { error, .. }) => match error {},
        }
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

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns an error if the index is out of range or applying an action fails while
    /// rebuilding the cached states; in that case, the action is not removed, but it may have
    /// been reordered past some of the actions it commutes with —
    /// [`RemoveActionError::ActionFailed`] carries its current index.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// # struct Set(usize, i32);
    /// # impl Action for Set {
    /// #  type State = Vec<i32>;
    /// #  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
    /// #   state[self.0] = self.1;
    /// #   Ok(state)
    /// #  }
    /// #  fn commute(&self, other: &Self) -> bool { self.0 != other.0 }
    /// #  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
    /// #   state[self.0] = previous_state[self.0];
    /// #  }
    /// # }
    /// let mut history = History::new(vec![0, 0]);
    /// history.push_action(Set(0, 1));
    /// history.push_action(Set(1, 2));
    /// assert_eq!(history.try_remove_action(0).unwrap(), Set(0, 1));
    /// assert_eq!(*history.last_state(), vec![0, 2]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*n*) time, where *n* is the number of actions after `index`. The consecutive
    /// actions after `index` that the removed action commutes with are shifted past with only
    /// *O*(log *n*) state clones, inversions, and applications; the actions after the first
    /// non-commuting one are re-applied in full.
    pub fn try_remove_action(&mut self, index: usize) -> Result<A, RemoveActionError<A>> {
        self.try_remove_action_with(index, &mut ())
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

    /// Removes and returns the action at `index` (the `index`th oldest action, which transforms
    /// the state at version `index` into the state at the following version). Later states are
    /// rebuilt as if the removed action had never been applied.
    ///
    /// Returns `None` and leaves the history unchanged if `index` is out of range.
    ///
    /// # Example
    ///
    /// ```
    /// # use history::*;
    /// # #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    /// # struct Set(usize, i32);
    /// # impl Action for Set {
    /// #  type State = Vec<i32>;
    /// #  fn apply(&self, mut state: Vec<i32>, _: &mut ()) -> Result<Vec<i32>, Self::Error> {
    /// #   state[self.0] = self.1;
    /// #   Ok(state)
    /// #  }
    /// #  fn commute(&self, other: &Self) -> bool { self.0 != other.0 }
    /// #  fn inverse(&self, previous_state: &Vec<i32>, state: &mut Vec<i32>) {
    /// #   state[self.0] = previous_state[self.0];
    /// #  }
    /// # }
    /// let mut history = History::new(vec![0, 0]);
    /// history.push_action(Set(0, 1));
    /// history.push_action(Set(1, 2));
    /// assert_eq!(history.remove_action(0), Some(Set(0, 1)));
    /// assert_eq!(*history.last_state(), vec![0, 2]);
    /// ```
    ///
    /// # Time complexity
    ///
    /// Takes *O*(*n*) time, where *n* is the number of actions after `index`. The consecutive
    /// actions after `index` that the removed action commutes with are shifted past with only
    /// *O*(log *n*) state clones, inversions, and applications; the actions after the first
    /// non-commuting one are re-applied in full.
    pub fn remove_action(&mut self, index: usize) -> Option<A> {
        self.remove_action_with(index, &mut ())
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

    impl Action for Option<TestAction> {
        type State = Vec<TestAction>;
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

    impl Action for SetSlot {
        type State = [u8; SLOT_COUNT];

        fn apply(
            &self,
            mut state: Self::State,
            _: &mut Self::Context,
        ) -> Result<Self::State, Self::Error> {
            state[self.slot()] = self.value;
            Ok(state)
        }

        fn commute(&self, other: &Self) -> bool {
            self.slot() != other.slot()
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

            // The history must be indistinguishable from one built from the remaining actions.
            let mut expected = History::new([0; SLOT_COUNT]);
            for action in &expected_actions {
                expected.push_action(*action);
            }
            prop_assert_eq!(history.cached_versions(), expected.cached_versions());
            for version in history.versions() {
                prop_assert_eq!(history.get_state(version), expected.get_state(version));
            }
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
