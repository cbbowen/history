Efficient undo history for non-invertible actions.

[Documentation](https://cbbowen.github.io/history)

# Motivation

An undo history can be implemented as a stack of application states, where each time the state changes, a new copy is pushed onto the stack. Undo is then as simple as popping the most recent state off the stack. The disadvantage of this approach is that it requires storing *O*(n) application states where n is the length of the undo history.

That's often prohibitive, so an undo history is commonly implemented instead as stack of actions applied to the application state, each of which must be invertible. This is the [command pattern](https://en.wikipedia.org/wiki/Command_pattern). Undo can then be implemented by popping the top action from the stack and applying its inverse to the application state[^1]. This still requires storing *O*(n) actions, but actions are typically _much_ smaller. The disadvantage of this approach is that it requires actions to be invertible. Often, this inverse must be manually implemented for each type of action.

This library provides a middle ground. It stores *O*(log n) states while still allowing pushing and popping actions in *O*(1) (amortized) time and a single action application, all without requiring invertible action. Furthermore, it also allows reconstructing the state at any point in the history in *O*(k + log n) time[^2].

[^1]: Observe that storing a stack of states is a special case of storing a stack of actions where each action is a replacement of the entire state.

[^2]: It is possible to get rid of the log n term in the running time of `get_state` but it's not generally relevant to the undo history application. If you have a use-case where it is, let me know.

# Usage

Implement the [`Action`] trait for your action type, add them to the history with [`History::push_action`], and retrieve the current stat with [`History::current_state`]. Remove the most recent action with [`History::pop_action`].

```rust
enum MyAction {
	Add(i32),
	Sub(i32),
}

impl history::Action for MyAction {
	type State = i32;
	fn apply(&self, state: i32) -> i32 {
		match self {
			MyAction::Add(a) => state + a,
			MyAction::Sub(a) => state - a,
		}
	}
}

let mut history = history::History::default();
history.push_action(MyAction::Add(710));
assert_eq!(*history.current_state(), 710);
history.push_action(MyAction::Sub(42));
assert_eq!(*history.current_state(), 668);
history.pop_action();
assert_eq!(*history.current_state(), 710);
```