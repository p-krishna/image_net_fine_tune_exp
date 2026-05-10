# image_net_fine_tune_exp
Experimental comparison of famous image classification models like AlexNet, VGG16.  These models are compared after they are retrained on German Traffic Sign Board Recognition dataset.  Retraining is done with all layers frozen except the last fully connected layer of all models.  This is purely a exploratory learning, not meant for any strict objective.

Kindly read TRAINING_GUIDE.md for instructions on how to perform retraining.  This repo contains code needed for training and making this whole experiment very convenient.  Please download data from kaggle [GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

I performed this on an average household CPU-only machine.  Training ran for several hours.  Code may need changes to the data preparation part based on dataset used and model training part to fit to GPU machines.  Kindly adhere to License explained in LICENSE.md