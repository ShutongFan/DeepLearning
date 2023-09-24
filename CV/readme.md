# DeepLearning_project

## Identifying Pneumonia by X-ray image based-on Deep learning

Nowadays, pneumonia, which is a contagious disease of the lungs, is seriously threatening people's health and normal life. In traditional scenarios, diagnosis is made artificially by doctors based on empiricism. But with the rapid development of deep learning and computer vision, there have been many studies proving their feasibility in practical applications. 

The aims of this report are designing a proficient model to detect pneumonia and its type. There are CNNs, transfer learning 3 pretrained models and their fine tuning in both of two tasks. But more in task 2 is transferring the best performance pretrained structure in task 1 into task 2 dataset. 

The result shows that transfer learning TF + VGG from task 1 outperforms other models. The evaluation metrics of accuracy, recall, precision, auc and F1 score is 92%, 79.1%, 99%, 93%, 88%. At the same time, it owns a lower time cost.

The report is in https://docs.google.com/document/d/1MxjLvlVpVJ-bny1IDefd3UUusb14-Y2jHT9rJg1YGMY/edit#

The chest X-ray images dataset used in this report is from: https://data.mendeley.com/datasets/rscbjbr9sj/3
