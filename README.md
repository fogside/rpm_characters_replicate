# Ready Player Me inspired posed 3D characters
## Stable Diffusion v1.5, ControlNet v1.1, Cog model 
[![Replicate](https://replicate.com/stability-ai/stable-diffusion/badge)](https://replicate.com/fogside/rpm_characters_concepts)

* [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)


This model generates 3D renders of a character from two angles. It uses ControlNet for keeping the pose consistency. It uses pre-trained model for producing Ready Player Me inspired characters with consistent face and back renders.

First, download the pre-trained weights:

    cog run script/download-weights 

Then, you can run predictions:

**Example 1**
``` bash
cog predict -i prompt="rebel-biker" -i body-type="masculine" -i negative_prompt="handbag"
```
**Example 2**
``` bash
cog predict -i prompt="cyberpunk goth model" -i body-type="feminine" -i negative_prompt="high contrast, dark shadows"
```

## Features
1. Long prompts
2. Emphasizing words with round brackets and lowering the impact of the word with square brackets (Automatic1111 style)