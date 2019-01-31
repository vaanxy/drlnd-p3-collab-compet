[//]: #	"Image References"
[image1]: ./images/maddpg.png	"MADDPG Scores"
# Report

In this project, Multi Agent Deep Deterministic Policy Gradient(MADDPG) has been implemented to solve the problem.

This report  will describe this learning algorithm, along with  the model architectures for neural networks and the chosen hyperparameters.

## Multi Agent Deep Deterministic Policy Gradient(MADDPG) 

### Actor Neural Network Architecture

The actor network mapping state to action

```python
Actor(
  (bn): BatchNorm1d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (bn2): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=128, out_features=128, bias=True)
  (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc4): Linear(in_features=128, out_features=64, bias=True)
  (bn4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc5): Linear(in_features=64, out_features=2, bias=True)
)
```



### Critic Neural Network Architecture

The critic network mapping (full_state, action1, action2) to Q-value.

I try to concat the states from both agents observations and combine actions from those two agents as the input of critic neural network.

~~~python
Critic(
  (bn): BatchNorm1d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc1): Linear(in_features=49, out_features=512, bias=True)
  (bn1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=512, out_features=256, bias=True)
  (bn2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc3): Linear(in_features=256, out_features=128, bias=True)
  (bn3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc4): Linear(in_features=128, out_features=64, bias=True)
  (bn4): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc5): Linear(in_features=64, out_features=1, bias=True)
)
~~~



### Hyper-parameters

- Replay Memory Size = 1e5
- Batch Size = 128
- GAMMA = 0.99
- TAU = 1e-3
- Actor Learning Rate = 1e-3
- Critic Learning Rate = 1e-4
- Noise Decaying Rate = 0.9999

### Plot of Rewards

![DDPG Scores][image1]

MADDPG solved the problem around 4000 episodes.


## Future Improvement

- Fine tuning hyper parameters to get better performance;
- Try to make the network deeper to  get better performance;
