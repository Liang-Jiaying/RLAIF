# Paper Presentation
<img width="700" alt="Screenshot 2023-11-01 at 6 18 36 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/70e483af-ab39-4760-abac-e142412f25a1">


# Overview
Five-minute overview 
providing context, 
stating the problem the paper is addressing, 
characterizing the approach, 
and giving a brief account of how the problem was addressed.

(Tips:   
!!! Overview, story is important. 
State the problem and how the paper solve it. (2 most important points)
Do not try to cover everything, it is not necessary (take a look at summary Claude, be deeper than that)

Highlight the important part.)


# Table of content
[Goal](#goal)  
[Background](#background)   
[Key Points](#key-points)
- [Point1](#part-1)
- [Point2](#part-2)
- [Pseudocode: ](#pseudocode)
  
[Resources](#references)  
[References](#references)  
[Contact Info](#contact-info)  

## Goal

## Background
To review: RLHF  
[Question 1] To guess: RLAIF

To review: SFT

## Key Points
### Part 1 
Go over steps of RLHF v.s. RLAIF, their different methodology
(Prepare a formal pseudocode description of the proposed model, indicate how it differs from previous models)

### Part 2 
Results from different prespective & techniques

### Part 3 
(6. qualitative analysis) & critical analysis
(Answer one or more of the following questions:   
**What was overlooked by the authors? [washed out]**   
What could have been developed further?   
Were there any errors?   
Have others disputed the findings?)


### Pseudocode
[Algorithm 1: Supervised Fine-tuning](#algorithm-1-supervised-fine-tuning)  
[Algorithm 2: Training Reward Model](#algorithm-2-training-reward-model)  
[Algorithm 3: Reinforcement Learning Fine Tuning](#algorithm-3-reinforcement-learning-fine-tuning)  

#### Algorithm 1: Supervised Fine-tuning
```
===============================
Algorithm 1: SupervisedFineTune
===============================
Input: 
  model: a pretrained language model
  train_data: a dataset of (input, target) pairs
    input: a text sequence
    target: a target text sequence
Output: 
  finetuned_model: a model finetuned on train_data

Hyperparameters:
  lr: learning rate
  num_epochs: number of training epochs

finetuned_model = model
for epoch in num_epochs:
  for (input, target) in train_data:
    logits = model(input)
    loss = cross_entropy(logits, target)
    grads = compute_grads(loss) 
    finetuned_model = update(finetuned_model, grads, lr)
return finetuned_model
```

#### Algorithm 2: Training Reward Model
```
=============================
Algorithm 2: TrainRewardModel
=============================
Input:
  model: a pretrained language model
  dataset: a dataset of inputs
Output:
  reward_model: a trained reward model  

Hyperparameters:
  num_samples: num response pairs to sample per input
  lr: learning rate
  num_epochs: number of training epochs

pairs = []
for input in dataset:
  for i in range(num_samples):
    response1, response2 = sample_model(model, input)
    pairs.append((input, response1, response2))

prefs = []  
for (input, response1, response2) in pairs:
  pref = get_llm_pref(response1, response2) 
  prefs.append((input, response1, response2, pref))

reward_model = RewardModel() 
for epoch in num_epochs:
  for (input, response1, response2, pref) in prefs:
    logits = reward_model(response1, response2)
    loss = cross_entropy(logits, pref)
    grads = compute_grads(loss)
    reward_model = update(reward_model, grads, lr)

return reward_model
```
#### Algorithm 3: Reinforcement Learning Fine Tuning
```
=======================
Algorithm 3: RLFineTune
=======================
Input:
  model: a pretrained language model
  reward_model: a trained reward model
  dataset: a dataset of inputs 
Output:
  finetuned_model: a model finetuned with RL

Hyperparameters:
  lr: learning rate
  num_epochs: number of training epochs

finetuned_model = model
for epoch in num_epochs:
  for input in dataset:
    response = sample_model(model, input)
    reward = reward_model(input, response)
    loss = rl_loss(model, input, response, reward) 
    grads = compute_grads(loss)
    finetuned_model = update(finetuned_model, grads, lr)

return finetuned_model
```


## References
1. Lee, H., Phatale, S., Mansoor, H., Lu, K., Mesnard, T., Bishop, C., ... & Rastogi, A. (2023). Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267.

## Resources
1. 

2. 
## Contact Info
Paper Presentor:  
Jiaying Liang (Email: jiaying.liang@vanderbilt.edu)
