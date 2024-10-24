## Folder Description

This folder contains ALITE and baseline source codes. The codes for computing biconnected components and strongly connected components are inherited from available open sources. 

This folder also contains separate folders for each benchmark and for embeddings given by each method. Download benchmarks and embeddings from [this link](https://drive.google.com/drive/folders/1yUgL8TjQievzp8zvmHLpa_ClNzc5mTmD?usp=sharing) and unzip them here.
Furthermore, [align_utilities/](align_utilities/) folder contains the code to generate the TURL and other embeddings for new dataset. Please run [alite_TURL_embedding_generation.ipynb](align_utilities/alite_TURL_embedding_generation.ipynb) notebook to generate the embeddings. 
Please find the config details on the [TURL GitHub repository](https://github.com/sunlab-osu/TURL?tab=readme-ov-file).
Also, TURL entity_vocab.txt file is provided by the authors [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EjZWRtslWX9CubQ92jlmNTgB74hxxXszy9BUaXG5OL5F-g?e=HN2qtD) (inside data folder).

The minimum example folder contains the tables used for minimum example in the paper. To compute FD of the minumum example tables, run [alite_fd.py](alite_fd.py) (using ALITE) or [pdelay_fd.py](pdelay_fd.py) (using BICOMNLOJ). Note that as the minimum examples do not have PK-FK relationship, [para_fd.py](para_fd.py) cannot compute their FD.
