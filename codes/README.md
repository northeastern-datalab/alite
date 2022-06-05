## Folder Description

This folder contains ALITE and baseline source codes. The codes for computing biconnected components and strongly connected components are inherited from available open sources. 

This folder also contains separate folders for each benchmark and for embeddings given by each method. Download benchmarks and embeddings from [this link](https://drive.google.com/drive/folders/1yUgL8TjQievzp8zvmHLpa_ClNzc5mTmD?usp=sharing) and unzip them here.
The minimum example folder contains the tables used for minimum example in the paper. To compute FD of the minumum example tables, run [alite_fd.py](alite_fd.py) (using ALITE) or [pdelay_fd.py](pdelay_fd.py) (using BICOMNLOJ). Note that as the minimum examples do not have PK-FK relationship, [para_fd.py](para_fd.py) cannot compute their FD.