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

## ◈ Primary contributions

• Style matching with adversarial alignment boosts robustness without key modalities.  

• Modality-specific adapters strengthen weak inputs for balanced segmentation results

• Lesion-aware priors reduce false positives and improve cross-modal consistency

• Ranks top in benchmark for missing-modality brain tumor segmentation performance.


## 🧗Proposed method
<br><br>
![](./imgs/fig1.png)
<br><br>

**Framework overview**  (A) **Missing-modality Sampling** — Generates 15 MRI modality combinations and leverages an Adapter Bank to compensate for absent inputs.  (B) **Knowledge-distillation Training** — Incorporates **BBDM**, **GARM**, and **LGRM**, with **GARM** applied exclusively to the student branch.

