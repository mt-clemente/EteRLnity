<a name="readme-top"></a>

# Solving Eternity 2 using reinforcement learning



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#the-agent">The agent</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#some-results">Results</a></li>
    <li><a href="#future-improvements">Future Improvements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<p align="center">
  <img width="300" src=https://github.com/mt-clemente/Avalam-DQN-Agent/assets/92925667/75bb1238-5308-42d5-b057-9476b70dadc2
 />
</p>

Eternity 2 is a board game that was created to be an unsolvable combinatorial 
optimization problem. The game goes as follow: place distinct square pieces with colored edges such that no touching edges are different colors. The full game board has a 16*16 size, which makes for a gigantic state space. To this day, this problem 
has not been solved, and people are still trying to find solutions with 
less and less conflicts.\
\
Most eternity related programs use heuristics / metaheuristics to get
results, but although these methods perform very well, they do not 
**learn** from their trajectory (even a tabu list could not realistically
remember every visited state). This project aims to try and  bring in some
learning, to see how well a deep neural network could perform with this task.


## The agent

To try and learn how this problem *behaves* without knowing the full 
solution, we use a reinforcement learning agent with **PPO**. The
particularity here, is that we do not use a CNN which would seem like
the most obvious way to go, because of the importance of spatial
structure in the task. Instead, we make two important choices:
 * The agent will be trained with the aiming of **solving** the problem,
 not just getting a better solution. This entails that when placing a piece, there is either one optimal orientation for a given (state, placing position). The agent then only has to learn a the order of the pieces and not their orientation. This greatly reduces the state space.
 * The task will be treated as a sequence processing task : the learning
 episode will start from an empty board and always fill it in the same order.

All of this justifies the usage of a **transformer** model, that will be 
used to predict the next tile to play, given a current state.


<!-- USAGE EXAMPLES -->
## Training and network settings
The training and network settings can be set using the 
`Network/param.py`. All parameters should be straightforward, except the
`POINTER` flag. The agent is using a transformer to predict the next tile to be played,
but there are a few ways we can handle the output of the transformer and interpret it
as a game piece (we call that the policy head). 
 * If the `POINTER` flag is set to false, a fully connected layer will take the current output token as input and output the policy directly.
 * If the `POINTER` flag is set to true, a pointer network architecture takes the connected layer's place. This has shown to reduce training speed and seemingly
 performance, but this has a major upside: the policy head does not have a fixed size
 anymore. This means that the model can be trained on problems of different sizes, without
 having to retrain a custom policy head.

The pointer network head performance issues are to take with a grain of salt, as the model
was trained on a laptop with a limited amount of computing power and training timer.

<!-- USAGE EXAMPLES -->
## Usage

### Requirements
To install the requirements simply run

  ```sh
   pip install -r requirements.txt
  ```

### Running the agent
The project is included with a makefile for simplicity of use:
 * You can train the model using x $\in${full,trivial,a,b,c,d,e}:
  ```sh
   make x
  ```

 * To test the model trained model, using X $\in${A,B...,F}, run:
  ```sh
   make gameX
  ```
 
 * Basic models have been implemented. A random baseline, a heuristic
 greedy agent, and a simple local search. Run them using:
  ```
   make random
   make heuristic
   make local
  ```
 

## Some results
As stated before, the computing power available to train the model was very limited
but the results are still encouraging: in a few thousands of episodes, the agent shows
clear signs of learning. This is using a basic transformer, with only up to 4 encoding
or decoding layers. 


<p align="center">
  <img src=results_rlco.png />
</p>

This is not performing that well compared to advanced metaheuristic algorithms, but it
seems more modular and adaptable. Once the metaheuristic algorithm is launched, it is
fixed, contrary to this agent. This model, although already more complicated than most
heuristic algorithms, still has many options for improvement.


<!-- ROADMAP -->
## Future improvements

- More extensive hyperparameter tuning
- Test the model on bigger GPUs, as VRAM was the limiting factor in model scaling.
- Test different types of policy heads.
- New, more stable transformer models (precisely TrXL,GTrXL) could offer good improvements
  on performance and training stability.
- Try new problem paradigms, for example but not only instead of choosing the piece given
a state and position, try taking a piece and letting the agent choose position.
