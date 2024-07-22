# DART2024 Dataset
We have uploaded the DART2024 dataset to the BaiduYun Disk (https://pan.baidu.com/s/1MeifqdtWE9QdnmvaXYKArg?pwd=lqrn)


# UIQA method in this article
![./Net](https://github.com/dart-into/DART2024/blob/main/network.png)
## Required Dataset
| Dataset   | Links                                                       |
| --------- | ----------------------------------------------------------- |
| SAUD      | https://github.com/yia-yuese/SAUD-Dataset     |
| UIED      | https://github.com/z21110008/UIF      |
| SOTA      | https://github.com/Underwater-Lab-SHU/IQA-Datatset      |
| UID2021   | https://github.com/Hou-Guojia/UID2021        |
| TID2013   | http://www.ponomarenko.info/tid2013.htm                              |
| KADID10K     | https://database.mmsp-kn.de/kadid-10k-database.html |
| LIVEC     | https://live.ece.utexas.edu/research/ChallengeDB/index.html |
## Usages

First, you need to download the required dataset and then modify the dataset path used in the file to ensure that the images can be accessed correctly.
### Pretraining  

Quality prior model training on TID2013 and KADID-10k database.
```
python MetaIQA_MultiNormal_On_TID2013_KADID.py
```
Quality prior model training on SOTA and UID2021 database.
```
python MetaIQA_MultiNormal_On_UID2021_SOTA.py
```
### Finetune 
Model fine-tuning on SAUD database.
```
python MetaIQA_MultiNormal_FineTune_SAUD.py
```
Model fine-tuning on UIED database.
```
python MetaIQA_MultiNormal_FineTune_UIED.py
```
Model fine-tuning on DART2024 database.
```
python MetaIQA_MultiNormal_FineTune_DART.py
```
