Efficient undo history for non-invertible actions.

[Documentation](https://cbbowen.github.io/history)

# Motivation

An undo history can be implemented as a stack of application states, where each time the state changes, a new copy is pushed onto the stack. Undo is then as simple as popping the most recent state off the stack. The disadvantage of this approach is that it requires storing *O*(n) application states where n is the length of the undo history.

That's often prohibitive, so an undo history is commonly implemented instead as a stack of actions applied to the application state, each of which must be invertible. This is the [command pattern](https://en.wikipedia.org/wiki/Command_pattern). Undo can then be implemented by popping the top action from the stack and applying its inverse to the application state[^1]. This still requires storing *O*(n) actions, but actions are typically _much_ smaller. The disadvantage of this approach is that it requires actions to be invertible. Often, this inverse must be manually implemented for each type of action.

This library provides a middle ground. It stores *O*(log n) states while still allowing pushing actions with a single application of an action and popping actions in *O*(log n) applications (amortized), without requiring invertible actions. Furthermore, it also allows reconstructing the state at any point in the history in *O*(k + log n) time[^2] where k is the age of the state.

[^1]: Observe that storing a stack of states is a special case of storing a stack of actions where each action is a replacement of the entire state.

[^2]: It is possible to get rid of the log n term in the running time of `get_state` but it's not generally relevant to the undo history application. If you have a use-case where it is, let me know.

# Usage

Implement the [`Action`] trait for your action type, add them to the history with [`History::push_action`], and retrieve the current state with [`History::last_state`]. Remove the most recent action with [`History::pop_action`].

```rust
enum MyAction {
	Add(i32),
	Sub(i32),
}

impl history::Action for MyAction {
	type State = i32;
	fn apply(&self, state: i32, _: &mut Self::Context) -> Result<i32, Self::Error> {
		Ok(match self {
			MyAction::Add(a) => state + a,
			MyAction::Sub(a) => state - a,
		})
	}
}

let mut history = history::History::default();
history.push_action(MyAction::Add(710));
assert_eq!(*history.last_state(), 710);
history.push_action(MyAction::Sub(42));
assert_eq!(*history.last_state(), 668);
history.pop_action();
assert_eq!(*history.last_state(), 710);
```

## Bounding the history

The history above is unbounded: it keeps every action forever, and only the *states* are held to *O*(log n). When that is too much, [`History::forget_actions`] folds the oldest actions into the initial state and drops them, which is the only way anything leaves the front.

It costs nothing to do. The cached states are spaced geometrically backwards from the most recent version, so folding stops at the oldest cached state within reach and there is no state to reconstruct and no action to apply — just the actions to drop. In exchange it folds in only as many as it can reach that way, so it can fall short of what you asked for by a bounded factor; call it again as the history grows and it keeps up.

```rust
# enum MyAction { Add(i32), Sub(i32) }
# impl history::Action for MyAction {
# 	type State = i32;
# 	fn apply(&self, state: i32, _: &mut Self::Context) -> Result<i32, Self::Error> {
# 		Ok(match self { MyAction::Add(a) => state + a, MyAction::Sub(a) => state - a })
# 	}
# }
let mut history = history::History::default();
for i in 0..1000 {
	history.push_action(MyAction::Add(i));
}
let version = history.last_version();

// Keep roughly the most recent hundred actions.
history.forget_actions(900);
assert!(history.actions().len() < 400);

// Whatever it dropped, the versions that remain are the versions they always were.
assert_eq!(history.last_version(), version);
assert_eq!(*history.last_state(), 499500);
assert_eq!(history.get_state(history::Version::default()), None);
```

Versions are absolute, counting every action ever applied. A [`Version`] held across a fold therefore never comes to mean a different state: either it still names the state it always named, or the operation that takes it says it is gone.