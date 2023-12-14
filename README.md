# Leveraging Physical Rules for Weakly Supervised Cloud Detection in Remote Sensing Images

## Abstract
Cloud detection plays a significant role in remote sensing image applications. 
Existing deep learning-based cloud detection methods rely on massive precise pixel-level annotations, which are time-consuming and expensive. 
To alleviate this problem, we propose a weakly supervised cloud detection framework that leverages physical rules to generate weak supervision for cloud detection in remote sensing images.
Specifically, a rule-based adaptive pseudo labeling algorithm is devised to adaptively annotate pseudo cloud masks based on cloud spectral properties without manual intervention.
Subsequently, these pseudo cloud masks are treated as weak supervision to optimize the heuristic cloud detection network for pixel-wise segmentation. 
Considering that clouds appear as complex geometric structures and nonuniform spectral reflectance, a deformable boundary refining module is designed to enhance the model capability of spatial transformation and activate precise boundaries from among translucent cloud regions.
Moreover, a harmonic loss is employed to recognize clouds with nonuniform spectral reflectance and suppress the interference of bright backgrounds.
Extensive experiments on the GF-1, Landsat-8, and WDCD datasets demonstrate that the proposed method achieves state-of-the-art results.




I will publish the whole source code after paper acception. If any question, please concat with us (email: yliucit@bjtu.edu.cn).



## Dataset
GF-1 Cloud and Cloud Shadow Cover Validation Data: [link](URL "http://sendimage.whu.edu.cn/en/mfc-validation-data")

Landsat-8 Cloud Cover Assessment Validation Data: [link](URL "https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data")

Weakly Supervised Cloud Detection Data: [link](URL "https://github.com/weichenrs/WDCD")

## Training Phase
Please run train_PHA_DBRM_GF1.py for model training.

## Testing Phase
Please run getPredictions.py to get model predictions. 
Then all patches are spliced to their original size by bac2src_output.py for evaluation. 




![abl_vis.png](./demo_graph/abl_vis.png)
<p align="center">Qualitative results of the GF-1 dataset images.</p>


## Cites
@ARTICLE{10180080,
  author={Liu, Yang and Li, Qingyong and Li, Xiaobao and He, Shuyi and Liang, Fengjiao and Yao, Zhigang and Jiang, Jun and Wang, Wen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Leveraging Physical Rules for Weakly Supervised Cloud Detection in Remote Sensing Images}, 
  year={2023},
  volume={61},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2023.3294817}}

# HeuristicCloudDetectionNetwork
