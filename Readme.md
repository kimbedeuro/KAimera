Killing Two Birds with KAimera: High-Accuracy, Low-Complexity Anomaly Detection for Industrial Control Systems through Knowledge Distillation
-------------
<span style="color:black;"> ***KAimera*** is a lightweight anomaly detection model, which applies a knowledge distillation approach. The model includes prediction values from 10 state-of-the-art anomaly detection models that have been evaluated on two public datasets. </span>

This repository contains the **KAimera** implemented in python and the dataset.

Getting started
-------------
We confirmed that ***KAimera*** runs on Ubuntu 18.04. 
* <span style="color:black;"> To access the source code, clone this repository using the following command: </span>

<pre><code><span style="color:black;"> git clone https://github.com/kimbedeuro/KAimera.git && cd KAimera </span>
</code></pre>

Build environment
-------------
<span style="color:black;"> Our implementation environment is as follows: </span>

* **Pytorch version 2.1.2**
  
* **TensorFlow 2.11.0**
  
* **Python 3.8.1**

If you want to see the whole environment list, you can confirm in <code>0.Gihub_SWaT.ipynb</code> file.  
(WADI and SWaT have the same implementation environment.)

Implementation
-------------
Once the repository and environment settings are finished, Kaimera can run as following commands:

<pre><code>#SWaT distillation
python Github_SWaT.py
  
#WADI distillation
python Github_WADI.py</code></pre>

If you want to see the code output result for SWaT and WADI directly,  
please check <code>0.Github_SWaT.ipynb</code> and <code>1.Github_WADI.ipynb</code>. 
