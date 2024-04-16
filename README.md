<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
-->


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://www.publicdomainpictures.net/en/view-image.php?image=372932&picture=artificial-intelligence">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

  <h1 align="center">Functional Brain Age</h1> <!-- NOTE: probably replace by the title of the paper -->

  <!-- <p align="center">
    Functiona Brain Age
    <br />
    <a href="https://github.com/pausz/fab-example"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/pausz/fab-example/issues">Report Bug</a>
    ·
    <a href="https://github.com/pausz/fab-example/issues">Request Feature</a>
  </p> -->
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#to-do">To-Do</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
     <li><a href="#citation">How to cite</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

![fba_figure1](https://github.com/brain-modelling-group/functional-brain-age/assets/39141919/25b49c8e-63ac-483a-9ed3-22fe3c2f5d47)


The code in this repository estimates the 'age' of a segment of EEG recording from a paediatric population. 
This EEG based estimate of age is referred to as the function brain age (FBA). 
This code relates to the following publication with a Python and Matlab version available.

**Inputs:**
- (1) EEG file in EDF format (a segment of N2 sleep).
- (2) The age of the subject of EEG recording (expressed in years).
- (3) The prediction algorithm used: 
    - training dataset: D1, D2 or D1&D2, 
    - number of channels: 2 or 18, 
    - algorithm type: neural netowrks or Gaussian process regression [Matlab version only]

**Outputs:** 
- (1) Functional Brain Age (expressed in years).
- (2) Centile from Growth Chart based on our data.
- (3) FBA corrected to align with growth chart.
- (4) Predicted age difference (PAD).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![python][python]][python-url]
* [![onnx][onnx-img]][onnx-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This script is an example of how you may use our trained network. To get a local copy up and running follow the steps below.

### Prerequisites
- We have tested this script on:
  Linux (Ubuntu 22.04 LTS), using python 3.9


### Installation

0. Clone the repo and navigate to the folder
   ```sh
   git clone https://github.com/brain-modelling-group/functional-brain-age
   cd functional-brain-age/
   ```
3. Create python environment, and activate 
   ```sh
   conda create -n py39-fba python=3.9
   conda activate py39-fba
   ```
4. Install required libraries to run the example
   ```sh
   pip install -e .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
To see how this works with the default dataset in Python, montage and trained NN, you can simply do:
   ```sh
   python demo.py 
   ```
You should see the following outputs:
   ```sh
          Empirical Age is 5 years.
  
          Predicted Functional Brain Age (FBA) is 4.987 years.
 
          Estimated centile from Growth Chart is 38.5%.

          Corrected Functional Brain Age (FBA) is [5.6094] years.

          Predicted Age Difference (PAD) is: [0.60938]
   ```

If you want to use a different NN with 2 channels you can do:
```
python demo.py --num_channels 2 --onnx_filename fba/data/onnx/D1D2_2ch_model_Opset12.onnx 
```


For usage in MATLAB (tested on 2022a), montage and trained NN, you can simply do the following, where 'path' is the full folder directory where the data is stored and 'filename' is the full name of the .edf file (e.g. 'subject1.edf') 
   ```sh
   [data,predictFBA,cFBA,centile,pad] = FBA(path,filename,5,'bipolar',[],'NN','D1_NN_18ch_model');
   ```
In the MATLAB command window, you should see a similar read out to the Python implementation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


Test runs of NN models executed within 10 seconds on either Python or MATLAB implementations on a Windows PC running an Intel(R) Xeon(R) W-1290P CPU @ 3.70GHz and 64 GB of RAM. 64-bit Operating system. 


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
- Nathan Stevenson - *nathan* dot *stevenson* _at_ *qimrberghofer* dot *edu* dot *au* 
- Kartik Iyer - *kartik* dot *iyer* _at_ *qimrberghofer* dot *edu* dot *au* 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- HOW TO CITE -->
## Citation
- If you use this code, please cite: <insert reference> 

Iyer, K.K., Roberts, J.A., Waak, M., Vogrin, S.J., Kevat, A., Chawla, J., Haataja, L.M., Lauronen, L., Vanhatalo, S. and Stevenson, N.J., 2024. A growth chart of brain function from infancy to adolescence based on EEG. EBioMedicine, 102.
(https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(24)00096-3/fulltext)


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The study was supported by an NHMRC Grant no. 2002135, the HUS Children’s Hospital/HUS diagnostic center research funds, Finnish Academy (335788, 332017), Finnish Pediatric Foundation (Lastentautiensäätiö), Sigrid Juselius Foundation
  
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://au.linkedin.com/in/nathan-stevenson-30a20923
[python]: https://camo.githubusercontent.com/3df944c2b99f86f1361df72285183e890f11c52d36dfcd3c2844c6823c823fc1/68747470733a2f2f696d672e736869656c64732e696f2f7374617469632f76313f7374796c653d666f722d7468652d6261646765266d6573736167653d507974686f6e26636f6c6f723d333737364142266c6f676f3d507974686f6e266c6f676f436f6c6f723d464646464646266c6162656c3d
[python-url]: https://www.python.org/
[onnx-url]: https://onnx.ai/
[onnx-img]: https://img.shields.io/badge/-ONNX-%20black?logo=onnx&logoColor=white&style=flat-square
