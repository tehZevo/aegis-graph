# Aegis Graph

## Components

### Sources
- have a state

### Nodes
- have a fixed state size
- are sources themselves
- have links to other sources
- calculate their state as an elementwise reduction of their links' outputs

### Links
- connect to sources
- are RL agents
- employ curiousity as the agent's intrinsic reward
- are "recurrent" by default (includes node's previous state as part of the agent's observation)

## TODO
- resolve `WARNING:absl:Skipping variable loading for optimizer 'SGD', because it has 7 variables whereas the saved optimizer has 1 variables.` if it is indeed an issue