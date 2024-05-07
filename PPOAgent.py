import numpy as np
import tensorflow as tf

class PolicyNetwork(tf.keras.Model):
    def __init__(self, input_shape, output_shape):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_shape, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)


class PPOAgent:
    def __init__(self, input_shape, output_shape, epsilon=0.2, gamma=0.99, lr_actor=0.0001, lr_critic=0.001):
        self.policy_network = PolicyNetwork(input_shape, output_shape)
        self.optimizer_actor = tf.keras.optimizers.Adam(lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(lr_critic)
        self.epsilon = epsilon
        self.gamma = gamma

    def get_action(self, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        logits = self.policy_network.call(state)
        action_probs = tf.nn.softmax(logits)
        action = np.random.choice(len(action_probs[0]), p=action_probs.numpy()[0])
        return action, action_probs[0, action].numpy()

    def get_value(self, state):
        state = np.expand_dims(state, axis=0)  # Add batch dimension
        value = self.policy_network.call(state)
        return value.numpy()[0, 0]

    def train_step(self, states, actions, old_probs, advantages, returns):
        states = np.array(states)
        actions = np.array(actions)
        old_probs = np.array(old_probs)
        advantages = np.array(advantages)
        returns = np.array(returns)

        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            logits = self.policy_network.call(states)
            probs = tf.nn.softmax(logits)
            action_masks = tf.one_hot(actions, probs.shape[1])
            selected_action_probs = tf.reduce_sum(action_masks * probs, axis=1)
            ratio = selected_action_probs / old_probs
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            value_preds = self.policy_network.call(states)
            value_loss = tf.reduce_mean(tf.square(returns - value_preds))

        gradients_actor = tape_actor.gradient(actor_loss, self.policy_network.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(gradients_actor, self.policy_network.trainable_variables))

        gradients_critic = tape_critic.gradient(value_loss, self.policy_network.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(gradients_critic, self.policy_network.trainable_variables))


def compute_advantages_and_returns(rewards, values, gamma=0.99):
    advantages = []
    returns = []
    advantage = 0
    returns.append(values[-1])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantage = delta + gamma * 0.95 * advantage
        advantages.append(advantage)
        returns.insert(0, advantage + values[t])
    advantages = np.array(advantages[::-1])
    returns = np.array(returns[:-1])
    return advantages, returns


def train_ppo_agent(agent, env, episodes=1000, max_steps=100):
    for _ in range(episodes):
        states = []
        actions = []
        rewards = []
        old_probs = []
        values = []

        state = env.reset()
        for _ in range(max_steps):
            action, prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_probs.append(prob)
            values.append(agent.get_value(state))

            state = next_state
            if done:
                break

        advantages, returns = compute_advantages_and_returns(rewards, values)
        agent.train_step(states, actions, old_probs, advantages, returns)
