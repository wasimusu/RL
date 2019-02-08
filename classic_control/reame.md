## Classical Control Problems using tabular Q-learning

I applied grid search over three values of alpha keeping all parameters fixed.

### Parameters
- epsilon = 0.04 for first 10k episodes and 0.01 for others
- discretization strategy : round(position, 1), round(velocity, 2)

### Plots
Average returns for every hundred episodes for different values of alpha (learning rate) for 30000 episodes

- a = 0.1, 

![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/0110.png)
![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/0130.png)

- a = 0.05

![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/00510.png)
![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/00530.png)

- a = 0.01

![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/00110.png)
![alt text](https://github.com/wasimusu/RL/blob/master/classic_control/plots/00130.png)
