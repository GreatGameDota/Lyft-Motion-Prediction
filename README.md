# Lyft Motion Prediction

My 122nd* place solution to the [Lyft Motion Prediction for Autonomous vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles) competition hosted on Kaggle by Lyft.

## Initial Thoughts

I joined rather late to this competition and then had to battle with memory leaks and errors but even still I feel like I did pretty well. This competition was interesting because you needed a long compute time which is not possible with online cloud environments like Colab. So I had to use my personal GPU and train for hours on end.

I would also like to shoutout @[louis925](https://www.kaggle.com/louis925) whose discussion posts about bug fixes[[1]](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/195259)[[2]](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/195936) made it possible for me to compete!

## Overview

My final solution was basically just the baseline model trained for 3.5 days. The final single model was simply a ResNet18 with a linear head dropout rate of 0.5. I trained it on 5% of the total training data randomly sampled.

## Models

Since training took so long and because I started experimenting late I couldn't try very many models. I only tried ResNet18 and ResNet34 variations of the baseline model. The final change that got my leaderboard score was adding a dropout layer in the linear head.

## Dataset

I didn't do anything fancy with the dataset. I just used the vanilla l5kit AgentDatasets and validated with the full validation.zarr chopped.

## Augmentation

Since there was such a large amount of data and I didn't dive deep into the data, so I had limited knowledge of what the data was, I didn't use any augmentation.

## Training

I used a simple pipeline for training:

- Adam optimizer with One cycle LR schedule
- Trained for 15 epochs with a batch size of 64
- Used the given neg multi log loss likelihood loss as the loss function
- Trained for 5% of the length of full_train.zarr each epoch and randomly sampled the entire train AgentDataset for every item.

Training the final model took 85.25 hours or 3.5 days on my local RTX2080 gpu.

## Ensembling/Blending

Did not ensemble or blend at all, just submitted one model once.

## Final Submission

Since everyone said the validation scheme matched so well with the leaderboard I only submitted my best model validation score wise.

```cpp
Validation: 20.39
Public LB: 19.89
Private LB: 20.54
```

## Final Thoughts

Nothing super important from this competition, just had fun and learned a lot as usual! I am content with my final placement as I just wanted to beat the baseline kernals, I was surprised I even got close to a bronze.

Previous Competition: [OSIC Pulmonary Fibrosis Progression](https://github.com/GreatGameDota/OSIC-Pulmonary-Fibrosis-Prediction)

Next Competition: [placeholder]
