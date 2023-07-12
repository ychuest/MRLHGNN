# License

Copyright (C) 2023 Li Peng (plpeng@hnu.edu.cn), Cheng Yang (yangchengyjs@163.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 3
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.



# MRLHGNN
MRLHGNN is effective tool for drug repositioning and we are thankful that [Gu et al.](https://www.sciencedirect.com/science/article/pii/S0010482522008356) have published part of their data which can be used directly.



# Environment Requirement
+ torch version (GPU) == 2.0.1
+ CUDA version == 12.0
+ numpy == 1.34.3
+ matplotlib == 3.5.1
+ dgl-cu118 == 1.1.0
+ pandas == 1.5.3
+ scikit-learn == 1.2.2
+ torch-cluster == 1.6.1+pt20cu118
+ torch-scatter == 2.1.1+pt20cu118
+ torch-sparse == 0.6.17+pt20cu118
+ torch-spline-conv == 1.2.2+pt20cu118
+ torchaudio ==2.0.2
+ torchvision == 0.15.2



# Model
+ load_data.py: Constructing heterogeneous graph.
+ SeHG.py: the core model proposed in the paper.



# Compare_models
 
* NTSIM (2017)
    * Proposed in [Predicting drug-disease associations based on the known association bipartite network](https://ieeexplore.ieee.org/abstract/document/8217698/), BIBM 2017.

* BNNR (2019)
    * Proposed in [Drug repositioning based on bounded nuclear norm regularization](https://doi.org/10.1093/bioinformatics/btz331), Bioinformatics 2019.

* HGIMC (2020)
    * Proposed in [Heterogeneous graph inference with matrix completion for computational drug repositioning](https://doi.org/10.1093/bioinformatics/btaa1024), Bioinformatics 2020.

* NIMCGCN (2020)
    * Proposed in [Neural inductive matrix completion with graph convolutional networks for miRNA-disease association prediction](https://doi.org/10.1093/bioinformatics/btz965), Bioinformatics 2020.

* LAGCN (2021)
    * Proposed in [Predicting drug–disease associations through layer attention graph convolutional network](https://doi.org/10.1093/bib/bbaa243), Briefings in Bioinformatics 2021.

* DRHGCN (2021)
    * Proposed in [Drug repositioning based on the heterogeneous information fusion graph convolutional network](https://doi.org/10.1093/bib/bbab319), Briefings in Bioinformatics 2021.

* DRWBNCF (2022)
    * Proposed in [A weighted bilinear neural collaborative filtering approach for drug repositioning](https://doi.org/10.1093/bib/bbab581), Briefings in Bioinformatics 2022.

* REDDA (2022)
    * Proposed in [REDDA: Integrating multiple biological relations to heterogeneous graph neural network for drug-disease association prediction](https://www.sciencedirect.com/science/article/pii/S0010482522008356), Computers in Biology and Medicine 2022.

* MilGNet (2022)
    * Proposed in [MilGNet: a multi-instance learning-based heterogeneous graph network for drug repositioning](https://ieeexplore.ieee.org/abstract/document/9995152/), BIBM 2022.



# Question
+ If you have any problems or find mistakes in this code, please contact with us: 
Cheng Yang: yangchengyjs@163.com ; Li Peng: plpeng@hnu.edu.cn
