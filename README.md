# Flype :flying_disc: : Foundational Models for Parameter Efficient Visual Language Understanding

## Update
Find our demo at http://flype.mtop.uk:7007, or at https://c8e1bf3ca4f04d6f8a.gradio.live

Our winning model written in [LAVIS](https://github.com/salesforce/LAVIS) for [CheckThat!](https://checkthat.gitlab.io/clef2023/task1/) is now available at [Flype-LAVIS](https://github.com/pengbohua/Flype-LAVIS)!
## Overview
To perform parameter-efficient visual language classification, we select the VQA foundation model, BLIP-2. 
As shown in the following figures, we propose parameter-efficient prompt-based learning, FLYPE, for visual language understanding in computational social science (CSS). 
The method consists of two stages: a cross-modal continuous prompt tuning stage and a prompt fusion stage.
<figure>
<img src="./assets/flype.png" style="width: 76%:"/>
    <figcaption style="text-align: center">Figure 1. The universal model architecture of FLYPE, cross-modal prompt tuning for large visual language models</figcaption>
</figure>

<figure>
<img src="./assets/prompt_fusion.png" style="width: 76%;"/>
    <figcaption style="text-align: center">Figure 2. Prompt fusion for multi-task learning</figcaption>
</figure>

## Run the code
Run the following code to find the hypers for your favorite task.
```python
python sweep.py
wandb agent  Entity/ProjectName/SweepID
```
