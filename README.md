## StressNet-Detecting-stress-in-thermal-videos
StressNet introduces a fast and novel algorithm of obtaining physiological signals and classify stress states from thermal videos. This repo contains ground-up write of all the components of StressNet. It is written in Python and powered by the pytorch deep learning framework.

### [**StressNet: Detecting Stress in Thermal Videos**](https://arxiv.org/pdf/2011.09540.pdf)
[Satish Kumar*](https://www.linkedin.com/in/satish-kumar-81912540/), [ASM Iftekhar](https://www.linkedin.com/in/a-s-m-iftekhar-86914b136/), [Michael Goebel](https://www.linkedin.com/in/mike-goebel-6331551bb/), [Tom Bullock](https://www.linkedin.com/in/tomwbullock/), [Mary H. MacLean](https://psych.ucsb.edu/people/researchers/mary-maclean), [Michael B. Miller](https://psych.ucsb.edu/people/michael-miller), [Tyler Santander](https://psych.ucsb.edu/people/researchers/tyler-santander), [Barry Giesbrecht](https://psych.ucsb.edu/people/faculty/barry-giesbrecht), [Scott T. Grafton](https://psych.ucsb.edu/people/faculty/scott-grafton), [B.S. Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath)

Official repository for our [**WACV 2021**](https://openaccess.thecvf.com/content/WACV2021/html/Kumar_StressNet_Detecting_Stress_in_Thermal_Videos_WACV_2021_paper.html)

<img src="overview.gif" width="700">

This repository includes:
* Source code for ISTI signal predictor with baseline model Resnet50
* Source code for Stress Detector
* Requirements file to setup the environment
* Training/Test code for both ISTI signal predictor and Stress Detector
* Example of training on your own dataset

The whole repo is structure follows the same style as written in the paper for easy reproducibility and easy to extend this work. If you use this research, please consider citing our paper (bibtex below)

## Citing
If this work is useful to you, please consider citing our paper:
```
@inproceedings{kumar2020stressnet,
  title={StressNet: Detecting Stress in Thermal Videos},
  author={Kumar, Satish and Iftekhar, ASM and Goebel, Michael and Bullock, Tom and MacLean, Mary H and Miller, Michael B and Santander, Tyler and Giesbrecht, Barry and Grafton, Scott T and Manjunath, BS},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={999--1009}
}
```

### Requirements
- Linux or macOS with Python >= 3.5
- Pytorch >= 1.4.0
- CUDA 10.1 or 10.0 or 9.2
- cudNN (compatible with CUDA)

### Installation
1. Clone the repository
2. Install dependencies
```
pip install -r requirements.txt
```
### ISTI detector
Running the ISTI detector code is quite simple. Follow the [REAMDME.md](link here)
```
src/README.md
```

### Stress detector from predicted ISTI signal
For detecting stress from thermal video, the ISTI detector needs to be run first. Then the predicted ISTI signal is used to predict the probability of stress experienced by an individual. Follow the [README.md](link here)
```
link here
```
