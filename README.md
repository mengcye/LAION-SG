# <p align="center"> LAION-SG: An Enhanced Large-Scale Dataset for Training Complex Image-Text Models with Structural Annotations </p>
*<p align="center">
  [Paper Link](https://arxiv.org/abs/2412.08580)*
</p>

*<p align="center">
  by Zejian li<sup>1</sup>, Chenye Meng<sup>1</sup>, Yize Li<sup>2</sup>, Ling Yang<sup>3</sup>, Shengyuan Zhang<sup>1</sup>, Jiarui Ma<sup>1</sup>, Jiayi Li<sup>2</sup>, Guang             Yang<sup>4</sup>, Changyuan Yang<sup>4</sup>, Zhiyuan Yang<sup>4</sup>, Jinxiong Chang<sup>5</sup>, Lingyun Sun<sup>1</sup>*
</p>

*<p align="center">
  <sup>1</sup>Zhejiang University  <sup>2</sup>Jiangnan Uniersity  <sup>3</sup>Peking University  <sup>4</sup>Alibaba Group  <sup>5</sup>Ant Group*
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

## Dataset
Our dataset has been published on Hugging Face. [Access the LAION-SG dataset](https://huggingface.co/datasets/mengcy/LAION-SG).
## Environment setup
The following commands are tested with Python 3.10 and CUDA 11.8.

Install required packages:

```
pip3 install -r requirements.txt
```
## Training
We provide a script for training `sdxl-sg` using the LAION_SG dataset. Use the following command to start `trainer_laion.py`:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes 8 trainer_laion.py
```

## Inference
We provide a simple inference script that allows generating images from the LAION-SG dataset.

1. **Download Pre-trained Weights**  
   Click [here](https://drive.google.com/file/d/1mdC3Np4KkV9V24K1gcyddsG5AIv5S0MT/view?usp=sharing) to download the pre-trained weights and place them in the root directory of the project.

2. **Create an Output Directory**  
   In the root directory of the project, create a folder named `output` to store the generated images:
   ```bash
   mkdir output
   ```

  Your project directory should look like this after step 2:
  ```LAION-SG/
  ├── configs/
  ├── output/
  ├── pics/
  ├── pretrained/
  ├── sgEncoderTraining/
  ├── baseline3_100.pt
  ├── LICENSE
  ├── README.md
  ├── requirements.txt
  ├── test_laion.py
  ├── trainer_laion.py
  ```
3. **Run the Inference Script**  
   Use the following command to perform inference:

   ```bash
   python test_laion.py
   ```
   
   The generated images will be saved in the `output/` folder as `{img_id}.jpg`, where `{img_id}` corresponds to the image ID from the LAION-SG dataset.
## Citation
```
@article{li2024laion,
  title={LAION-SG: An Enhanced Large-Scale Dataset for Training Complex Image-Text Models with Structural Annotations},
  author={Li, Zejian and Meng, Chenye and Li, Yize and Yang, Ling and Zhang, Shengyuan and Ma, Jiarui and Li, Jiayi and Yang, Guang and Yang, Changyuan and Yang, Zhiyuan and others},
  journal={arXiv preprint arXiv:2412.08580},
  year={2024}
}
```

