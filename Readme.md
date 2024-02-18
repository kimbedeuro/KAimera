Killing Two Birds with KAimera: High-Accuracy, Low-Complexity Anomaly Detection for Industrial Control Systems through Knowledge Distillation
-------------
<span style="color:black;"> KAimera is a lightweight anomaly detection model, which applies a knowledge distillation approach. The model includes prediction values from 10 state-of-the-art anomaly detection models that have been evaluated on two public datasets. </span>

This repository contains the KAimera implemented in python and the dataset.

Getting started
-------------
<span style="color:black;"> clone the repo </span>

<pre><code><span style="color:black;"> git clone https://github.com/kimbedeuro/KAimera.git && cd KAimera </span>
</code></pre>

Environment
-------------
<span style="color:black;"> Our implementation environment is as follows: </span>

* **Pytorch version 2.1.2**
  
* **TensorFlow 2.11.0**
  
* **Python 3.8.1**

Run
-------------
<pre><code>#SWaT distillation
python Github_SWaT.py
  
#WADI distillation
python Github_WADI.py</code></pre>
 
Citation
-------------
If you find this repo or our work useful for your research, please consider citing the paper.
**(This part will be updated after being accepted.)**


<pre><code>@inproceedings{-,
  title={Killing Two Birds with KAimera: High-Accuracy, Low-Complexity Anomaly Detection for Industrial Control Systems through Knowledge Distillation},
  author={Anonymous},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}</code></pre>
