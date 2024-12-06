# LAION-SG: An Enhanced Large-Scale Dataset for Training Complex Image-Text Models with Structural Annotations
<p align="center">
  paper link
  </p>

<p align="center">
  by *Zejian li<sup>1</sup>, Chenye Meng<sup>1</sup>, Yize Li<sup>2</sup>, Ling Yang<sup>3</sup>, Shengyuan Zhang<sup>1</sup>, Jiarui Ma<sup>1</sup>, Jiayi Li<sup>2</sup>, Guang Yang<sup>4</sup>, Changyuan Yang<sup>4</sup>, Zhiyuan Yang<sup>4</sup>, Jinxiong Chang<sup>4</sup>, Lingyun Sun<sup>1</sup>*
  </p>

<p align="center">
  *<sup>1</sup>Zhejiang University  <sup>2</sup>Jiangnan Uniersity  <sup>3</sup>Peking University  <sup>4</sup>Alibaba Group*
  </p>
  
![teaser](https://github.com/mengcye/LAION-SG/blob/main/pics/figure1_teaser.png)



## Abstract
Recent advances in text-to-image (T2I) generation have shown remarkable success in producing high-quality images from text. 
However, existing T2I models show decayed performance in compositional image generation involving multiple objects and intricate relationships.
We attribute this problem to limitations in existing datasets of image-text pairs, which lack precise inter-object relationship annotations with prompts only. 
To address this problem, we construct LAION-SG, a large-scale dataset with high-quality structural annotations of scene graphs (SG), which precisely describe attributes and relationships of multiple objects, effectively representing the semantic structure in complex scenes.
Based on LAION-SG, we train a new foundation model SDXL-SG to incorporate structural annotation information into the generation process. 
Extensive experiments show advanced models trained on our LAION-SG boast significant performance improvements in complex scene generation over models on existing datasets. 
We also introduce CompSG-Bench, a benchmark that evaluates models on compositional image generation, establishing a new standard for this domain. 

## Environment setup
The following commands are tested with Python 3.10 and CUDA 11.8.

Install required packages:

```
pip3 install -r requirements.txt
```
## Training
ckpt link
## Citation

