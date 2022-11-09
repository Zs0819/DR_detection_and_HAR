# Team10
- Shuai Zhang (st169622)
- Peng GU (st169623)

# Work distribution
- Diabetic Retinopathy:
  
  - Shuai ZHANG: model architecture, training, hyperparameter optimization, transfer learning, ensemble learning and multi-class classification.
  
  -  Peng GU: input pipeline(binary classification & multi-class classification), metrics, evaluation, data augmentation and deep visualization.
- Human Activity Recognition:

  - Shuai ZHANG: input pipeline, metrics and evaluation.

  - Peng GU: model architecture, training and hyperparameter optimization.

- Portability:
  
  - Shuai ZHANG & Peng GU: relative path, flags and different configuration files.

# How to run the code

## Installation
- Python 3
- Tensorflow2.6-gpu
- numpy
- matplotlib
- opencv-python
- gin-config
- absl-py
- ray
- scikit-learn
- seaborn
- pandas
- tensorflow_datasets
- glob

## Dataset Preparation
Datasets:
- IDRID:  
  - Introduction: The Indian Diabetic Retinopathy Image Dataset (IDRID) is a dataset of retinal fundus images. 
    It is a part of the database of "Diabetic Retinopathy: Segmentation and Grading Challenge". Images in the 
    IDRID are categorized into two parts, retinal images with and without signs of diabetic retinopathy. Each 
    image is associated with a ground truth respectively label, that is the severity grade of diabetic retinopathy.
    
  - Download link:  `https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2667521_Datasets`
  
- HAPT:
  - Introduction:Human Activities and Postural Transitions Dataset (HAPT) consists of six basic activities and six postural 
    transitions between the static activities. The static activities include standing, sitting and lying. The dynamic 
    activities include walking, walking downstairs and walking upstairs. Stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit,
    stand-to-lie, and lie-to-stand are the classes for postural transitions between the static activities.
    
  - Download link:  `https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2667521_Datasets`

Location:
- download the corresponding datasets, put [IDRID](https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2667521_Datasets) dataset under the folder
`diabetic_retinopathy/datasets/`, put [HAPT](https://ilias3.uni-stuttgart.de/goto.php?target=wiki_2667521_Datasets) dataset under the folder `human_activity_recognition/datasets/`.

TFRecords: 

- under the folder `diabetic_retinopathy/input_pipeline/`, run the following command to convert IDRID dataset into TFRecords
`python3 create_smaller_tfrecord.py`. 

- under the folder `human_activity_recognition/input_pipeline/`, run the following command to convert IDRID dataset into TFRecords
`python3 create_tfrecords.py`.


## Diabetic Retinopathy

- Binary classification:
  
  Training:
  - train vgg model:
  
    `python3 main.py --mode train --model_name vgg --num_classes 2`
  
  - train resnet model:
    
    `python3 main.py --mode train --model_name resnet --num_classes 2`
  - train transfer model:
  
    `python3 main.py --mode train --model_name transfer --num_classes 2`
  
  Test:
  - evaluate vgg model:
  
    `python3 main.py --mode evaluate --model_name vgg --num_classes 2`
  
  - evaluate resnet model:
  
    `python3 main.py --mode evaluate --model_name resnet --num_classes 2`
  
  - evaluate transfer model:
  
    `python3 main.py --mode evaluate --model_name transfer --num_classes 2`
  
  - evaluate ensemble model:
  
    `python3 ensemble.py --num_classes 2`
  
  Deep visualization:
  - visualization for vgg model:
  
    `python3 deep_visualization.py --model_name vgg --num_classes 2`
  
    visualization images are located under the folder: `diabetic_retinopathy/checkpoint/vgg/visualization/`
  
  - visualization for resnet model:
  
    `python3 deep_visualization.py --model_name resnet --num_classes 2`
  
    visualization images are located under the folder: `diabetic_retinopathy/checkpoint/resnet/visualization/`


- Multi-classes classification:

  Training:
  
    `python3 main.py --mode train --model_name transfer --num_classes 5`

  Test:

    `python3 main.py --mode evaluate --model_name transfer --num_classes 5`

## Human Activity Recognition
Training:
- train LSTM model:

  `python3 main.py --mode train --model_name lstm`
- train GRU model:

  `python3 main.py --mode train --model_name gru`

Test:
- evaluate LSTM model:

  `python3 main.py --mode evaluate --model_name lstm`
- evaluate GRU model:

  `python3 main.py --mode evaluate --model_name gru`

# Results
## Diabetic Retinopathy
- Binary classification:
  - result table:
    
    | Model | Accuracy | Sensitivity/Recall | Specificity | Precision | F1 score |
    | :---: | :---: | :---: | :---: | :---: | :---: |
    | vgg | 87.4% | 0.86 | 0.90 | 0.93 | 0.89 |
    | resnet | 87.4% | 0.81 | 0.97 | 0.98 | 0.89 |
    | transfer | 84.5% | 0.88 | 0.79 | 0.88 | 0.88 |
    | ensemble-voting | 88.3% | 0.84 | 0.95 | 0.96 | 0.90 |
    | ensemble-averaging | 88.3% | 0.84 | 0.95 | 0.96 | 0.90 |

  - confusion matrix:
    - [confusion matrix test on vgg model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/confusion_matrix.png)
    - [normalized confusion matrix test on vgg model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/normalized_confusion_matrix.png)
    - [confusion matrix test on resnet model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/confusion_matrix.png)
    - [normalized confusion matrix test on resnet model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/normalized_confusion_matrix.png)
    - [confusion matrix test on transfer model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/transfer/confusion_matrix.png)
    - [normalized confusion matrix test on transfer model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/transfer/normalized_confusion_matrix.png)
    - [normalized confusion matrix test on ensemble model with averaging method](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/ensemble/ensemble_averaging_cm.png)
    - [normalized confusion matrix test on ensemble model with voting method](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/ensemble/ensemble_voting_cm.png)
  
  - ROC curve: 
    - [ROC curve test on vgg model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/roc.png)
    - [ROC curve test on resnet model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/roc.png)
    - [ROC curve test on transfer model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/transfer/roc.png)
  
  - deep visualization:
  
    | Model | Image | Grad-CAM | Guided Grad-CAM |
    | :---: | :---: | :---: | :---: |
    | vgg | [original image-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/original_1.png) | [Grad-CAM-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/gradcam1.png) | [Guided Grad-CAM-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/guided_gradcam1.png) |
    | vgg | [original image-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/original_2.png) | [Grad-CAM-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/gradcam2.png) | [Guided Grad-CAM-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/guided_gradcam2.png) | 
    | vgg | [original image-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/original_3.png) | [Grad-CAM-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/gradcam3.png) | [Guided Grad-CAM-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/vgg/visualization/guided_gradcam3.png) | 
    | resnet | [original image-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/original_1.png) | [Grad-CAM-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/gradcam1.png) | [Guided Grad-CAM-1](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/guided_gradcam1.png) | 
    | resnet | [original image-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/original_2.png) | [Grad-CAM-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/gradcam2.png) | [Guided Grad-CAM-2](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/guided_gradcam2.png) | 
    | resnet | [original image-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/original_3.png) | [Grad-CAM-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/gradcam3.png) | [Guided Grad-CAM-3](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/2classes/resnet/visualization/guided_gradcam3.png) | 

  
  - location: Test results are located under `../experiments` folder named by runtime.
  
- Multi-classes classification:
  - result table:
  
    | Model | Accuracy |
    | :---: | :---: | 
    | transfer | 42.7% |
  
  - confusion matrix:
    
    - [confusion matrix test on transfer model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/5classes/confusion_matrix.png)
    - [normalized confusion matrix test on transfer model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/DR/5classes/normalized_confusion_matrix.png)

  
  - location: Test results are located under `../experiments` folder named by runtime.

## Human Activity Recognition
- result table:

    | Model | Accuracy |
    | :---: | :---: | 
    | LSTM | 92.4% | 
    | GRU | 93.7% |

- confusion matrix:
  - [confusion matrix test on LSTM model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/HAR/LSTM/abs_cm_LSTM.png)
  - [normalized confusion matrix test on LSTM model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/HAR/LSTM/cm_LSTM.png)
  - [confusion matrix test on GRU model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/HAR/GRU/abs_cm_GRU.png)
  - [normalized confusion matrix test on GRU model](https://github.tik.uni-stuttgart.de/iss/dl-lab-21w-team10/blob/master/result/HAR/GRU/cm_GRU.png)

- location: Test results are located under `../experiments` folder named by runtime.
