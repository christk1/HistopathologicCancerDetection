# HistopathologicCancerDetection
Histopathologic Cancer Detection in histopathologic scans of lymph node sections.

This project uses the dataset from [here](https://www.kaggle.com/c/histopathologic-cancer-detection/data)
and for performance evaluation i made an 80/20 split of the train(folder) dataset for train/validation.

The performance is 90+% accuracy on validation set.

If you want to make a kaggle submission, you can add the remaining 20%(validation) to the training set, retrain
and hopefully gain much better results.

Validation is the blue line, training is the orange one.

![alt text](imgs/train_acc.png) ![alt text](imgs/val_acc.png) ![alt text](imgs/loss.png) 
