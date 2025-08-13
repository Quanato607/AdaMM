<h1 align="center"> N&nbsp;O&nbsp;&nbsp;M&nbsp;⬢&nbsp;D&nbsp;A&nbsp;L&nbsp;I&nbsp;T&nbsp;Y&nbsp;&nbsp;L&nbsp;E&nbsp;F&nbsp;T&nbsp;&nbsp;B&nbsp;E&nbsp;H&nbsp;I&nbsp;N&nbsp;D</h1>


<div align="center">

[![](https://img.shields.io/github/stars/Quanato607/MST-KDNet)](https://github.com/Quanato607/AdaMM)
[![](https://img.shields.io/github/forks/Quanato607/MST-KDNet)](https://github.com/Quanato607/AdaMM)
[![](https://img.shields.io/badge/project-page-red.svg)](https://github.com/Quanato607/AdaMM)
[![](https://img.shields.io/badge/arXiv-2403.01427-green.svg)](https://arxiv.org/abs/2030.12345)
</div>

This implementation of **No Modality Left Behind: Adapting to Missing Modalities via Knowledge Distillation for Brain Tumor Segmentation**. 

## 🎥 Visualization for Implementation on Software

<div align="center">
<img src="https://github.com/Quanato607/MST-KDNet/blob/main/imgs/implementation.gif" width="90%">
</div>

## ◈ Related Works

(a) **Data Generation** — An external generator synthesizes absent modalities, creating a complete four-channel input for the segmentation model.  

(b) **Feature Generation** — The network learns to hallucinate modality-specific features internally when inputs are missing.  

(c) **Sample Retrieval** — Retrieves training cases from modality-matched cohorts to substitute absent scans before segmentation.  

(d) **Robustness Enhancement** — Trains with random modality dropout to segment directly from available scans.  

(e) **Multi-task Learning** — An auxiliary decoder reconstructs absent modalities (red dashed arrows) while the main branch outputs the segmentation mask.  

(f) **Knowledge Distillation** — A full-modality teacher guides a partial-modality student via feature and prediction alignment, improving accuracy under incomplete inputs.


<p align="center">
  <img src="./imgs/fig5.png" alt="Figure 3" width="90%">
</p>
<p align="center">
  <img src="./imgs/fig6.png" alt="Figure 3" width="90%">
</p>

## 🧗Proposed method
<br><br>
![](./imgs/fig1.png)
<br><br>

**Framework overview**  (A) **Missing-modality Sampling** — Generates 15 MRI modality combinations and leverages an Adapter Bank to compensate for absent inputs.  (B) **Knowledge-distillation Training** — Incorporates **BBDM**, **GARM**, and **LGRM**, with **GARM** applied exclusively to the student branch.

## 📝 Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## 🔥 Training

To train our model in the paper, run this command:

```train
python train.py
```

>📋 Before training, specify the data set and training configuration using the config.xml file

## 📃 Evaluation

To evaluate our model in the paper, run this command:

```eval
python eval.py
```

<br><br>
![](./imgs/fig2.png)
<br><br>

## 🚀 Results of Performance

<br><br>
![](./imgs/fig9.png)
<br><br>

## 🚀 Results of Comparision Experiment

<br><br>
![](./imgs/fig8.png)
<br><br>

## 🚀 Results of Ablation Experiment

<br><br>
![](./imgs/fig7.png)
<br><br>
