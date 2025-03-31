# test_repository
creating repository for test
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Defining the Environment
class OffloadingEnvironment:
    def __init__(self):
        self.state_size = 3  # (CPU, Bandwidth, Latency)
        self.action_size = 3  # (Local Processing, Offload, Forward)
        self.state = np.random.rand(self.state_size) * 100  # Initial random state

    def step(self, action):
        # Define rewards: Higher reward for lower latency and efficient offloading
        if action == 0:  # Local Processing
            reward = -self.state[0]  # CPU load penalty
        elif action == 1:  # Offload to Edge Cloud
            reward = -self.state[1]  # Bandwidth cost
        else:  # Forward Task
            reward = -self.state[2]  # Latency cost

        # Update state randomly (simulating changing network conditions)
        self.state = np.random.rand(self.state_size) * 100
        return self.state, reward

# Define the Deep Q-Learning Agent
class DQNAgent:
    def __init__(self):
        self.memory = []
        self.gamma = 0.95  # Discount Factor
        self.epsilon = 1.0  # Exploration Probability
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation="relu", input_shape=(3,)),
            keras.layers.Dense(24, activation="relu"),
            keras.layers.Dense(3, activation="linear")  # Output: Q-values for each action
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.choice([0, 1, 2])  # Random Action
        return np.argmax(self.model.predict(state.reshape(1, -1)))

    def replay(self):
        if len(self.memory) < 10:
            return

        batch = random.sample(self.memory, 10)
        for state, action, reward, next_state in batch:
            target = reward + self.gamma * np.max(self.model.predict(next_state.reshape(1, -1)))
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

        # Reduce exploration probability
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the Model
env = OffloadingEnvironment()
agent = DQNAgent()

for episode in range(500):
    state = env.state
    for _ in range(10):  # Run 10 steps per episode
        action = agent.act(state)
        next_state, reward = env.step(action)
        agent.remember(state, action, reward, next_state)
        state = next_state

    agent.replay()

print("Training complete!")
