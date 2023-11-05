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
[Background](#play_or_pause_button-background)   
[Key Points](#3-key-points)
- [Point1](#point-1-human-evaluators-strongly-prefer-rlhf-and-rlaif-summaries-over-the-supervised-fine-tuned-sft)
- [Point2](#part-2-for-the-task-of-summarization-rlaif-summarise-is-a-viable-alternative-to-rlhf-summaries-that-does-not-depend-on-human-annotation)
- [Pseudocode: ](#pseudocode)
  
[Resources](#references)  
[References](#references)  
[Contact Info](#contact-info)  

## Goals & Questions
1.
2. What things we need to be careful about when we are doing RLAIF?  

## :play_or_pause_button: Background
[REVIEW] What is Supervised Fine-Tuning (SFT)?  
[REVIEW] What is Refinforcement Learning (RL)?  

To help refresh the memory:   
**Supervised Fine-Tuning**:  
Principle: Supervised fine-tuning is a continuation of the **supervised learning process** where a pre-trained model is further trained (fine-tuned) on a new dataset that is usually smaller and more specific than the data used for pre-training.  
Data: It **requires labeled data**. **Each input in the training set has a corresponding output label**, and the model is trained to predict these labels as accurately as possible.
Objective: The model's parameters are adjusted to **minimize a loss function**, which measures the difference between the predicted labels and the true labels.  
Use Case: It is typically used when you have a target task that is similar but not identical to the task the model was originally trained on. For example, you might fine-tune a language model pre-trained on a large corpus of text to perform sentiment analysis on product reviews.  

**Reinforcement Learning**:  
Principle: RL is a type of machine learning where **an agent learns to make decisions by performing actions in an environment and receiving rewards** (or penalties) based on the outcomes of these actions.  
Data: It does **not require labeled data** in the traditional sense. Instead, the learning process is driven by the rewards that the agent receives from the environment.  
Objective: The goal is to **learn a policy—a mapping** from states of the environment to actions—**that maximizes the cumulative reward over time**.  
Use Case: RL is often used in control tasks (like robotics or video games), where the correct **actions are not known ahead of time and must be discovered through trial and error**.  

<!-- Supervised fine-tuning and reinforcement learning (RL) are different machine learning paradigms used for enhancing the performance of models.
In essence, supervised fine-tuning adjusts a model to perform better on a specific task with labeled data, while reinforcement learning is about learning from interaction with an environment to maximize some notion of cumulative reward. 
They are tools used for different purposes and are not similar in their core methodologies, although they can sometimes be used complementarily, such as using supervised learning to provide a starting point for an RL system (i.e., pre-training with supervised learning and then fine-tuning with RL).-->

## 3 Key Points
### Point 1: Human evaluators strongly prefer RLHF and RLAIF summaries over the supervised fine-tuned (SFT)
<img width="700" alt="Screenshot 2023-11-05 at 1 05 23 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/09937b00-d838-4a65-a07e-46651b0ff3b1">  

> (Win Rates: Win rate is a performance metric commonly used in competitive games and other situations where two or more entities compete against each other, and the outcome is a win, loss, or draw. Win rate is calculated as the ratio of the number of wins to the total number of games or matches played. The formula for win rate is:  
> $$\text{Win Rate} = \frac{\text{Number of Wins}}{\text{Total Number of Games}}$$  
> Win rate is often expressed as a percentage, where a win rate of 0.5 or 50% indicates that the entity has won half of the games played.)  

<!-- The differences in win rates between RLAIF vs. SFT and RLHF vs. SFT are not statistically significant. -->

### Part 2: For the task of summarization, RLAIF summarise is a viable alternative to RLHF summaries that does not depend on human annotation
<img width="985" alt="Screenshot 2023-11-05 at 1 36 52 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/3cff366d-8309-44e8-aff0-086b46f6af05">

<img width="960" alt="Screenshot 2023-11-05 at 1 46 58 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/f99f2e29-1081-4343-8716-c6b9f65ce530">

**The techniques used to generate preference labels with an LLM:**  

**1. Addressing Position Bias**     
  :interrobang: Problem: do have position bias   
  | Model Size | Position | % the Position Preferred |
  | :---         |     :---:      |          ---: |
  | PaLM 2 L   | 1st      | 94%    |
  | PaLM 2 S     | 2nd       | 91%     |
  | PaLM 2 XS     | 2nd     | 99%     |
  
  :white_check_mark: Solution: swap positions  <!-- position bias is inversely proportional to model size -->    
  <img width="407" alt="Screenshot 2023-11-05 at 1 20 17 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/9d10f8c8-cb0e-4dc4-bda1-460de2dcbd81">  

**2. Chain-of-thought Reasoning**    
  <img width="816" alt="Screenshot 2023-11-05 at 1 40 18 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/25323c27-9d8a-4afa-a3dc-5603edf07a7f">  

  **OpenAI + COT 0-shot:**  
  <img width="791" alt="Screenshot 2023-11-05 at 1 43 24 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/5e792c4e-042c-4806-b011-9d6d53484bdd">

  **OpenAI + COT 1-shot:**  
  <img width="800" alt="Screenshot 2023-11-05 at 1 43 48 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/263b23d8-05f0-4644-a3c8-8fac510bfce8">

  **Result:**  
  <img width="366" alt="Screenshot 2023-11-05 at 1 47 50 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/5e39de2d-3c82-4a42-8b4e-b644dbc4988b">   
  > AI Labeler Alignment: refers to the process of ensuring that the labels generated by an AI system are aligned with the intended task or objective.
  <img width="529" alt="Screenshot 2023-11-05 at 1 51 29 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/e885a9de-f2bf-421b-b861-f696ac76d780">

**3. Self-Consistency (non-zero decoding temperature)**    
  > Self-consistency refers to the property that a model's predictions or outputs are consistent with each other. This can mean that the model's predictions are consistent across different inputs that should lead to the same output, or that the model's predictions are consistent with its internal representations and learned knowledge.  
  <img width="413" alt="Screenshot 2023-11-05 at 1 56 34 AM" src="https://github.com/Liang-Jiaying/RLAIF/assets/111295386/86a4a7a7-baf6-489e-be05-78c380aed271">   
  
  > Note: 1, 4, and 16 samples represent 2, 8, and 32 inferences given our position de-biasing technique


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

2. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., ... & Lowe, R. (2022). Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35, 27730-27744.

## Resources
1. [Aligning language models to follow instructions by OpenAI](https://openai.com/research/instruction-following)

2. [Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems](https://arxiv.org/abs/2203.02155)

3. [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html)


## Contact Info
Paper Presentor:  
Jiaying Liang (Email: jiaying.liang@vanderbilt.edu)
