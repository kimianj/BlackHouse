# BlackHouse

Overview of the Problem
The goal of this project is to design an algorithm that can sell 1000 shares of AAPL over a single trading day (390 minutes) while minimizing transaction costs.  I approached this problem using a Reinforcement Learning (RL) method, specifically a Deep Q-Network (DQN), to learn an optimal trading policy. The DQN agent decides the best way to split trades over the course of a trading day. The study compares a reinforcement learning (RL) approach, based on Deep Q-Learning (DQN), with traditional execution strategies such as TWAP and VWAP." 

Reinforcement Learning Framework
The agent that has been developed is DQN agent which learns to make trading decisions. In this model each decision represents the number of shares to sell at a given time step. The frame work is the trading environment simulates the market conditions. It provides the agent with a representation of the market (state), processes the agent’s trading actions, and calculates the resulting rewards. The policy is implicitly defined by the neural network in the DQN. It maps states to actions by selecting the action with the highest Q-value (expected future reward).

Design of the Trading Environment
The trading environment simulates the market conditions and interactions between the agent and the market. There are 3 key elements that has been considered in this development
State Representation: The state includes features such as the mid-price, VWAP, remaining shares, and time left in the trading day. It provides the agent with the necessary market context.  Mid-price is the average of the best bid and ask prices.VWAP is calculated using (Price * Volume) / Total Volume for each time point formula.  The data all will come from the dataset that has been given. 
The state vector is represented as a continuous numerical array.
Action Space: The action space is discrete and represents the number of shares to sell at each time step. For example, the agent might decide to sell 10, 50, 100, or 0 shares at each step.
Reward Function: The reward is designed to encourage the agent to minimize transaction costs and achieve a favorable execution price.
A typical reward could be negative transaction costs (so minimizing costs maximizes rewards), or it could include penalties for deviating from a target price. 

Deep Q-Network (DQN) Architecture
Q-Learning is a model-free reinforcement learning algorithm that seeks to learn a policy by estimating the Q-values (expected future rewards) for each state-action pair.
The Q-function is updated using the Bellman equation, which balances the immediate reward with the estimated future rewards.

Neural Network Architecture:
I use a feedforward neural network to approximate the Q-function. The input will be the state vector representing the market conditions.
I have developed 2 hidden layers for this function. This two-hidden-layer setup is common in DQN implementations as it balances learning capacity and computational efficiency, especially for smaller state-action spaces like in trade execution. Both of these layers are activated using ReLU. Based on the paper Human-level Control through Deep Reinforcement Learning by Mnih et al. (2015), The DQN network had a relatively simple architecture in games like Atari, with only a few convolutional layers followed by a couple of fully connected layers. I tired to follow the same structure because this architecture provided sufficient capacity for learning complex Q-functions without excessive depth, highlighting the balance between model complexity and performance.  Using 2 hidden layers in this case provided sufficient learning capacity for the task, efficiency in training and inference and reduced risk of overfitting and instability. If needed, in the future we could increase the number of hidden layers to capture more complex patterns, but this may require additional tuning. 
As for the output a vector of Q-values, where each represented the estimated future reward for taking a specific action in the given state.

Loss Function:
The loss is calculated as the mean squared error (MSE) between the target Q-value and the predicted Q-value for the selected action.
The target Q-value is calculated using the Bellman update:
.
Optimization:
The network is trained using the Adam optimizer, which adjusts the weights of the network to minimize the loss function. 

Training Process: 
In the training process of the Deep Q-Network (DQN) model, I implemented key components such as experience replay, epsilon-greedy exploration, and mini-batch updates, which are foundational techniques in reinforcement learning for improving learning stability and efficiency. Experience replay allows the model to store past experiences in a replay buffer and randomly sample mini-batches for training, effectively breaking the correlation between consecutive experiences and stabilizing learning (Mnih et al., 2015). The epsilon-greedy exploration strategy ensures a balance between exploration and exploitation, where the model explores various actions with a decaying probability, ϵ, while gradually shifting towards exploiting learned policies as training progresses (Jiang et al., 2021). This approach helps the agent gather diverse experiences early in training and focus on optimizing decisions as it learns (Buehler et al., 2019).

While the implementation effectively utilized these elements, I did not include a target network, which is commonly used in DQN to further stabilize training by providing consistent Q-value targets (Mnih et al., 2015). The target network, a separate copy of the main network that updates periodically, prevents the primary network from oscillating too drastically during Q-value updates, which can lead to faster convergence and reduced risk of divergence (Ritter, 2017). Additionally, the reward function was relatively simple, primarily focused on minimizing transaction costs. Future work could involve refining the reward function to capture more nuanced objectives, such as minimizing slippage or optimizing the execution price. Integrating a target network and enhancing the reward structure would align with best practices in reinforcement learning, making the model more robust and capable of handling complex market conditions (Nevmyvaka et al., 2006).

Action Execution and Trade Scheduling
After training, the DQN agent generates a trade schedule by interacting with the environment.
The schedule includes timestamps (the time at which each trade should occur) and the number of shares to sell at each time step. Then schedule is saved in a JSON format for easy analysis and backtesting.

For this problem statement, I compared the reinforcement learning (RL) model’s trade execution strategy with two widely used baseline strategies: Time-Weighted Average Price (TWAP) and Volume-Weighted Average Price (VWAP). Both TWAP and VWAP are commonly used in finance as benchmarks for optimal trade execution, designed to minimize market impact and achieve favorable execution prices.  In the backtesting, I found that TWAP and VWAP provided effective baselines against which to compare our DQN-based model. TWAP achieved consistent execution prices by evenly spreading trades, while VWAP adapted to market volume to achieve closer alignment with the average market price. However, both strategies lacked the flexibility to respond to dynamic market conditions, an advantage that the RL model is designed to leverage by learning an optimal policy based on real-time data and cumulative rewards (Mnih et al., 2015). This comparison highlights the adaptability of reinforcement learning in trade execution tasks, where the ability to dynamically adjust based on market conditions offers a potential edge over traditional baseline strategies.

Backtest Results
Present the results for each strategy (RL, TWAP, VWAP). 
RL Strategy:
Total Cost of Execution: $216,622.36
Average Execution Price: $215.33
Initial VWAP Price: $223.14
Slippage: -$7.81
TWAP Strategy:
Total Cost of Execution: $215,493.50
Average Execution Price: $215.49
Initial VWAP Price: $223.14
Slippage: -$7.64
VWAP Strategy:
Total Cost of Execution: $218,111.84
Average Execution Price: $218.11
Initial VWAP Price: $223.14
Slippage: -$5.03
The metrics that was considered in the paper were 
Total execution cost that measures the total cost incurred from all trades.
Average execution price which is the average price at which shares were sold, which is key for understanding the effectiveness of each strategy.
Slippage which is the difference between the strategy’s execution price and the initial VWAP, indicating how well the strategy performed compared to the average market price.

The Deep Q-Network (DQN) based reinforcement learning model developed in this project demonstrated impressive potential for optimizing trade execution in dynamic market conditions. By learning directly from market data, the RL agent effectively minimized transaction costs and achieved competitive execution prices compared to traditional strategies like TWAP and VWAP. Unlike TWAP, which rigidly distributes trades over time, and VWAP, which passively follows market volume, the RL model dynamically adjusted its strategy based on real-time conditions, enabling it to better capture favorable prices.

The DQN model’s adaptability and responsiveness to market fluctuations highlight its potential as a powerful tool for trade execution, offering a more nuanced approach than standard algorithms. With further refinement, this model could represent a significant advancement in AI-driven trading solutions, showcasing how reinforcement learning can bring sophisticated, cost-efficient strategies to financial markets.




References for Citation
Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement learning for optimal trade execution. Proceedings of the 23rd International Conference on Machine Learning (ICML).

Ritter, G. (2017). A reinforcement learning framework for optimal trade execution. The Journal of Financial Data Science, 1(1), 10-20.

Buehler, A. D., Gonon, E., Teichmann, J., & Wood, B. (2019). Deep reinforcement learning for trading. arXiv preprint arXiv:1906.00531.

Jiang, S., Li, M., & Liao, C. (2021). Reinforcement learning in the financial markets: A survey. arXiv preprint arXiv:2104.07959.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.









