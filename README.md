# BBSEA 👴👍👶🏻
Unofficial implementation of [BBSEA: An Exploration of Brain-Body Synchronization for Embodied Agents](https://arxiv.org/abs/2402.08212).

![BBSEA Overview](./imgs/BBSEA_Overview.png)

## Motivation of This Repo
Here is the original repo for the [official BBSEA implementation](https://github.com/yangsizhe/bbsea/tree/main). The official implementation of BBSEA is based on the codebase of [Scaling Up and Distilling Down: Language-Guided Robot Skill Acquisition](https://github.com/real-stanford/scalingup), which adopts the simulation environment of [Mujoco](https://mujoco.org/).

However, I've observed a growing preference among Embodied AI researchers for using the simulation environments of [Maniskill](https://maniskill.readthedocs.io/en/latest/) and [Sapien](https://sapien.ucsd.edu/) due to their high rendering quality.

Therefore, I decided to implement BBSEA in Maniskill3 (a beta version of Maniskill). This effort is driven not only by my desire to contribute to the open-source community but also to enhance my engineering skills in conducting Embodied AI research.

## Ondoing Checklist

Now I have only relicated the Perception modules and LLM-Interaction modules of BBSEA, the primitive action APIs in Maniskill3 are still on doing...
