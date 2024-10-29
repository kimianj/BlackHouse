
import pandas as pd
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import json

file_path = "/content/AAPL_Quotes_Data.csv"
aapl_data = pd.read_csv(file_path)

# Convert the 'timestamp' column
aapl_data['timestamp'] = pd.to_datetime(aapl_data['timestamp'])

#missing values
missing_values = aapl_data.isnull().sum()

# Calculate the mid-price as the average of the best bid and ask prices
aapl_data['mid_price'] = (aapl_data['bid_price_1'] + aapl_data['ask_price_1']) / 2

# Calculate the bid-ask spread
aapl_data['bid_ask_spread'] = aapl_data['ask_price_1'] - aapl_data['bid_price_1']

# Calculate the VWAP
# VWAP = (Price * Volume) / Total Volume for each time point
aapl_data['vwap'] = (aapl_data['bid_price_1'] * aapl_data['bid_size_1'] +
                     aapl_data['ask_price_1'] * aapl_data['ask_size_1']) / (
                     aapl_data['bid_size_1'] + aapl_data['ask_size_1'])

print(aapl_data.head(10))

aapl_data['timestamp'] = pd.to_datetime(aapl_data['timestamp'])

# Environment for trading execution
class TradingEnvironment:
    def __init__(self, data, initial_shares=1000, initial_cash=0):
        self.data = data
        self.initial_shares = initial_shares
        self.shares_remaining = initial_shares
        self.cash = initial_cash
        self.current_step = 0
        self.done = False

    def reset(self):
        """Resets the environment for a new episode."""
        self.shares_remaining = self.initial_shares
        self.cash = 0
        self.current_step = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        """Returns the current state."""
        state = [
            self.shares_remaining,
            390 - self.current_step,
            self.data.iloc[self.current_step]['mid_price'],
            self.data.iloc[self.current_step]['bid_ask_spread'],
            self.data.iloc[self.current_step]['vwap'],
        ]
        return np.array(state)

    def step(self, action):
        """
        Executes a trade action.
        Action: number of shares to sell.
        """
        if self.done:
            return self._get_state(), 0, self.done

        current_price = self.data.iloc[self.current_step]['mid_price']
        trade_size = min(action, self.shares_remaining)
        execution_price = current_price - (self.data.iloc[self.current_step]['bid_ask_spread'] / 2)

        self.shares_remaining -= trade_size
        self.cash += trade_size * execution_price

        self.current_step += 1
        self.done = (self.current_step >= len(self.data) - 1) or (self.shares_remaining <= 0)

        reward = -abs(execution_price - self.data.iloc[self.current_step]['vwap']) * trade_size

        return self._get_state(), reward, self.done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
      #Builds a neural network model for approximating Q-values.
      model = Sequential()
      model.add(Dense(16, input_dim=self.state_size, activation='relu'))
      model.add(Dense(16, activation='relu'))
      model.add(Dense(self.action_size, activation='linear'))
      model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
      return model


    def remember(self, state, action, reward, next_state, done):
        """Stores the experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #Selects action based on the current state.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        #Trains the model using experiences from memory.
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = TradingEnvironment(aapl_data)
state_size = len(env.reset())
action_size = 10
agent = DQNAgent(state_size, action_size)
episodes = 5
batch_size = 32

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    #full day
    for time in range(390):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state


        if time % 50 == 0:
            print(f"Episode: {e+1}, Time Step: {time}, Action: {action}, Shares Remaining: {env.shares_remaining}")

        if done:
            print(f"Episode {e+1}/{episodes} - Cash Earned: {env.cash}, Shares Remaining: {env.shares_remaining}")
            break

    # Train the model at the end of the episode if enough experiences are stored
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

def generate_trade_schedule(agent, env):
    """Generates a trade schedule using the trained agent and saves it as a JSON file."""
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    trade_schedule = []

    while not env.done:
        action = agent.act(state)
        next_state, _, done = env.step(action)
        state = np.reshape(next_state, [1, state_size])

        trade_schedule.append({
            "timestamp": str(env.data.iloc[env.current_step]['timestamp']),
            "share_size": int(action)  # Ensure `action` is converted to a standard `int`
        })

    # JSON format
    with open('trade_schedule.json', 'w') as f:
        json.dump(trade_schedule, f, indent=4)
    print("Trade schedule saved as trade_schedule.json")

# Generate the schedule
generate_trade_schedule(agent, env)

def calculate_twap(shares, total_minutes=390):
    """TWAP strategy - evenly distribute shares across the trading day."""
    shares_per_minute = shares // total_minutes
    remaining_shares = shares % total_minutes  # Calculate leftover shares

    # Distribute evenly across the time steps
    twap_schedule = [{"timestamp": str(env.data.iloc[i]['timestamp']), "share_size": shares_per_minute}
                     for i in range(total_minutes)]

    # Distribute any remaining shares evenly among the first 'remaining_shares' time steps
    for i in range(remaining_shares):
        twap_schedule[i]["share_size"] += 1

    return twap_schedule

def calculate_vwap(data, shares):
    """VWAP strategy - distribute shares based on volume profile."""
    total_volume = data['vwap'].sum()
    trade_schedule = []

    # Calculate shares to allocate per volume share, scaled to the target share count
    volume_shares = [data.iloc[i]['vwap'] / total_volume for i in range(len(data))]
    scaled_shares = [int(round(volume_share * shares)) for volume_share in volume_shares]

    # Create the initial trade schedule with the scaled shares
    for i in range(len(data)):
        trade_schedule.append({
            "timestamp": str(data.iloc[i]['timestamp']),
            "share_size": scaled_shares[i]
        })

    # Adjust to ensure the total shares match exactly the target shares
    total_allocated_shares = sum(scaled_shares)
    share_difference = shares - total_allocated_shares

    # Distribute any leftover shares evenly across the schedule
    if share_difference > 0:
        for i in range(share_difference):
            trade_schedule[i % len(trade_schedule)]['share_size'] += 1
    elif share_difference < 0:
        for i in range(abs(share_difference)):
            # Avoid reducing shares below zero
            if trade_schedule[i % len(trade_schedule)]['share_size'] > 0:
                trade_schedule[i % len(trade_schedule)]['share_size'] -= 1

    return trade_schedule

shares_to_sell = 1000

twap_schedule = calculate_twap(shares_to_sell)
print("TWAP Schedule:")
print(twap_schedule[:5])


vwap_schedule = calculate_vwap(env.data, shares_to_sell)
print("\nVWAP Schedule:")
print(vwap_schedule[:5])

print("Sample VWAP values:")
print(env.data['vwap'].head())

# Calculate the total number of shares allocated by each schedule
twap_total_shares = sum([trade['share_size'] for trade in twap_schedule])
vwap_total_shares = sum([trade['share_size'] for trade in vwap_schedule])

print(f"Total shares allocated by TWAP: {twap_total_shares}")
print(f"Total shares allocated by VWAP: {vwap_total_shares}")

target_shares = 1000
print(f"Target shares to sell: {target_shares}")
print(f"TWAP matches target: {twap_total_shares == target_shares}")
print(f"VWAP matches target: {vwap_total_shares == target_shares}")

def backtest_strategy(schedule, env):
    """Simulates the execution of a trading schedule and calculates total cost."""
    total_cost = 0
    total_shares_sold = 0

    for trade in schedule:
        timestamp = trade["timestamp"]
        share_size = trade["share_size"]

        market_data = env.data[env.data['timestamp'] == timestamp]

        if not market_data.empty:
            execution_price = market_data.iloc[0]['mid_price']
            trade_cost = share_size * execution_price
            total_cost += trade_cost
            total_shares_sold += share_size

    avg_execution_price = total_cost / total_shares_sold if total_shares_sold > 0 else 0
    return total_cost, avg_execution_price

# Generate the schedules
shares_to_sell = 1000

rl_schedule = json.load(open('trade_schedule.json'))
twap_schedule = calculate_twap(shares_to_sell)
vwap_schedule = calculate_vwap(env.data, shares_to_sell)


rl_total_cost, rl_avg_execution_price = backtest_strategy(rl_schedule, env)
twap_total_cost, twap_avg_execution_price = backtest_strategy(twap_schedule, env)
vwap_total_cost, vwap_avg_execution_price = backtest_strategy(vwap_schedule, env)

def print_backtest_results(strategy_name, total_cost, avg_execution_price, initial_vwap):
    """Prints the results of the backtest."""
    print(f"{strategy_name} Backtest Results:")
    print(f"Total Cost of Execution: ${total_cost:.2f}")
    print(f"Average Execution Price: ${avg_execution_price:.2f}")
    print(f"Initial VWAP Price: ${initial_vwap:.2f}")
    slippage = avg_execution_price - initial_vwap
    print(f"Slippage: ${slippage:.2f}")
    print("-" * 40)


initial_vwap = env.data['vwap'].mean()

print_backtest_results("RL Strategy", rl_total_cost, rl_avg_execution_price, initial_vwap)
print_backtest_results("TWAP Strategy", twap_total_cost, twap_avg_execution_price, initial_vwap)
print_backtest_results("VWAP Strategy", vwap_total_cost, vwap_avg_execution_price, initial_vwap)

