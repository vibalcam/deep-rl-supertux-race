# Deep Reinforcement Learning SuperTux Race Kart

Deep Reinforcement Learning AI to play the SuperTuxKart race game.

## General

- Train in multiple scenarios

## Process only using lighthouse

- Pretrain CNN autoencoder, grayscale and RGB color
- Get data no drift running aim controller baseline (150 runs)
  - Noise (0,0.5,1 multipliers of 0.1 10)
    - 30 no noise
    - 60 noise (0.05, 2.5)
    - 60 noise (0.1, 5)
- Multiple trained models, overfit training by a lot --> 
  - try to generate more data by including noise to the aim controller
  - try warm start learning rate
- Solved calculating rewards to go (1)
- Train with aim controller without drift --> 
  - rewards of 132, 746 steps on lighthouse (fixed vel 0.5 no drift)
  - rewards of 200, 679 steps on lighthouse (fixed vel 1 no drift)
  - *successfully learns to drive but cannot take quick turns*
- Get data with/without drift running aim controller baseline
  - Drift enabled/disabled
  - Noise (0,0.5,1 multipliers of 0.1 10)
    - 15 no noise
    - 30 noise (0.05, 2.5)
    - 30 noise (0.1, 5)
- From best no drift model, enable drift and train over with/without drift data --> 
  - rewards of 203, 675 steps on lighthouse (fixed vel 0.5)
  - rewards 289, 590 steps on lighthouse (fixed vel 1)
  - beats baseline which obtains 273, 605 steps on lighthouse
  - beats ia 0 which obtains 243, 636 steps on lighthouse
  - *successfully learns to drive and drift so it can take quick turns, but does not control acceleration*
  - *beats the baseline from which it has learned*
- Train model from scratch using CNN encoder for drift, steer and acceleration
  - mean rewards of 280, 598 steps on lighthouse

## Problems

1. Not calculating reward to go correctly, it should be final - cum

## Ideas

- Train with adafactor optimizator
- augment with combined agent


## Environment

[PySuperTuxKart](https://github.com/philkr/pystk)
