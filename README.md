# simulations
Simulation framework for studying data collection biases using simulated student-item interactions.

## Components
### simulations.py
Python source code file with all the logic. Feel free to look around, make modifications, and extend.

### scenarios.json
The settings for simulations in JSON format. It is an easy way to change parameters independently of code.

### requirements.txt
A set of Python packages required to run the simulations. Can be easily installed with pip (Python package manager). 

### data
A folder where a simulation log is stored by default. Storing logs helps with reproducibility of results and saves time.

### plots
A folder where various plots, e.g., learning curves, are stored.

## How to run the simulations
1. Install all the required packages using `pip install -r requirements.txt`
2. Run `python simulations.py`

## How to modify
If you wish to play with parameters, then *scenarios.json* is the place to look. You can easily create a new scenario by modifying and combining already existing scenarios.

If the parameter modification is not enough, you can create your own initial distributions, student learning modes, item selection mechanisms and much more in *simulations.py*. Most actions happen by chaining methods that modify the simulation state. There is no limit on the number of chained methods. They can also pass results between each other.
