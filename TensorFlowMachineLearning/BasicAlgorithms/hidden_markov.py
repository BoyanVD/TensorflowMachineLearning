"""
    In order to create a hidden markov model we need to following main components :
     1. States - in each model we have a finite set of states. These states are hidden, which means that we dont
     observe them directly.

     2. Observations - each state has observation associated with it based on a probability distribution.

     3. Transitions - each state will have a probability defining the likelyhood of transitioning to a
     different state.

     With hidden markov models we predict future events based on past events.
"""

import tensorflow as tf
import tensorflow_probability as tfp

"""
    Initialization of constants :
"""
PROBABILITY_OF_COLD_FIRST_DAY = 0.8
PROBABILITY_OF_HOW_FIRST_DAY = 0.2

PROBABILITY_OF_HOT_DAY_AFTER_HOT_DAY = 0.8
PROBABILITY_OF_COLD_DAY_AFTER_HOT_DAY = 0.2

PROBABILITY_OF_HOT_DAY_AFTER_COLD_DAY = 0.3
PROBABILITY_OF_COLD_DAY_AFTER_COLD_DAY = 0.7

HOT_DAY_MEAN_TEMPERATURE = 15.
COLD_DAY_MEAN_TEMPERATURE = 0.

HOT_DAY_STD = 10.
COLD_DAY_STD = 5.

"""
    Creating the distributions :
"""
tensorflow_distribution = tfp.distributions
initial_distribution = tensorflow_distribution.Categorical(probs=[PROBABILITY_OF_COLD_FIRST_DAY, PROBABILITY_OF_HOW_FIRST_DAY])
transition_distribution = tensorflow_distribution.Categorical(
    probs=[[PROBABILITY_OF_COLD_DAY_AFTER_COLD_DAY, PROBABILITY_OF_HOT_DAY_AFTER_COLD_DAY],
            [PROBABILITY_OF_COLD_DAY_AFTER_HOT_DAY, PROBABILITY_OF_HOT_DAY_AFTER_HOT_DAY]]
)

observation_distribution = tensorflow_distribution.Normal(
    loc=[COLD_DAY_MEAN_TEMPERATURE, HOT_DAY_MEAN_TEMPERATURE],
    scale=[COLD_DAY_STD, HOT_DAY_STD]
)

"""
Creating the model :
"""
model = tensorflow_distribution.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7
)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())
