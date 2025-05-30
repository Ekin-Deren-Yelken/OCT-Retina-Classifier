# OCT-Retina-Classifier

An 8-layer convolutional neural network was built to classify retinal OCT images into four "classes": CNV, DME, DRUSEN, and NORMAL. The model was designed from scratch using the Keras Sequential API and follows a classic CNN structure optimized for feature extraction and classification. Each component was selected to progressively refine the image features and reduce overfitting.

## Neural Network Architecture

The following architecture was chosen for its ability to learn from scratch using the Kermany2018 dataset, which contains over 80,000 labeled grayscale OCT images. The network is compact enough to train on CPU or a single GPU

#### Layer 1 (32 filters 3x3 kernal, ReLU)
Detects low-level patterns such as edges, curves, and textures. Uses ReLU activation to keep on positive feature responses. Input shape (HEIGHT_SIZE, WIDTH_SIZE) set to accept grayscale images

#### Layer 2 (MaxPooling2D Layer (2x2))
Reduces dimensionality and computational load by downsampling.Only the most important features are selected in each region of the image to perserve key spatial information to help mitigate overfitting and make the representation more compact.

#### Layer 3 Conv2D Layer (64 filters, 3x3 kernal, ReLU)
Applies more filters to detect complex and abstract features formed by compinining low-level features in layer 1. Richer representations for differentiating subtle differences between OCT classes.

#### Layer 4 (MaxPooling2D Layer (2×2))
Reduces spatial size to prepare for dense classification, compressing the learned features and retaining the most "striking" characteristics

#### Layer 5 (Flaten Layer)
Converts 3D output volume from convolutiopn int a 1D vector to pass into layers, allowing the model to treat all learned features equally in the later stage.

#### Layer 6 (Dense Layer 128, ReLU)
Combines features from previous layers to form a high-level interpretation of the image. ReLU here to help the model learn complex decision boundries.

#### Layer 7 (Dropout Layer)
Regularization technique: Randomly deactivate half the neurons during training to reduce reliance on a couple pathways, reducing overfitting and improving generalization.

#### Layer 8 (Dense Output Layer (4 units, softmax))
outputs class (CNV, DME, DRUSEN, NORMAL) probabilities across the categories using softmax activation. Selects the class with the highest probability as the prediction.

## Results
While training the model, an early stop mechanism is used to stop training and automatically choose a number of epochs. This saves, time, avoids overfitting, and keeps the best weights. Two metrics on the validation dataset are used loss and accuracy. 

Loss reflects both how right and how confident the model is. This makes it highly sensitive. If this metric starts going up, the model may be overfitting. It is expected that this will stop the training model sooner.

Accuracy reflects classification correctness. This is more ideal for a medical software. When accuracy stops improving, training is stopped. Accuracy between epochs can change in larger steps and is usually better for cleaner datasets.

Both metrics were tested as seen in the image below.

![image](https://github.com/user-attachments/assets/ed54b681-77d0-420f-a771-972615e38172)

Early stopping on validation accuracy performs significantly better. val_accuracy trained with 9 epochs vs. only 5 with val_loss. This is expected as val_loss is more sensitive to confidence and may stop when predictions are correct but uncertain. While val_accuracy is expectd to perform better, in a clinical setting where certainty matters, it could prove to be unreliable.

#### Training Curves

![image](https://github.com/user-attachments/assets/8f017a35-c686-43f9-a938-b4fe575db947)


Training curves confirm better generalization with val_accuracy showing a steady increase in both train and val accuracy and no signs of overfitting even up to 9 epochs. val_loss stopped too early due to noise, losing out on a chance to improve. 

- Validation loss can sometimes plateau early due to softmax output variance, especially on imbalanced or complex datasets.

### Classification

In terms of classification, 4 metrics are used.
Precision: Out of all predictions made for a given class, how many were actually correct? (Measures False Positives)
- High precision means the model rarely mislabels other classes as this one.


Recall: Out of all actual examples of a class, how many did the model correctly identify? (Measures False Negatives)
- High recall means the model doesn’t miss real cases.


F1 Score: The harmonic mean of precision and recall. Accounts for both false positives and false negatives.


Support: The number of actual occurrences of each class in the test set. It indicates the relative weight or importance of that row’s metrics.


#### Metric Monitoring Success

![image](https://github.com/user-attachments/assets/d842dfe2-3ab6-4744-ad29-f17b347db66c)

The model using early stopping on validation accuracy significantly outperforms the one monitored on validation loss:
- The best model achieved an accuracy of 87% versus 62% from the loss monitor.
- F1-scores are on average ~0.86 indicating a meaningful improvement in both precision and recall compared to 0.56 in the loss monitor.
- All four classes show stronger and more consistent performance in the accuracy-monitored model.


#### Confusion Matrix

![image](https://github.com/user-attachments/assets/106b4342-d4bc-43c4-b736-f961987db480)


#### Class by Class Insights
| Class      | Early Stopping on Loss                                                  | Early Stopping on Accuracy                           |
| ---------- | ----------------------------------------------------------------------- | ---------------------------------------------------- |
| **CNV**    | High recall (0.99) but low precision (0.54) > overpredicting CNV        | Balanced and accurate: Precision 0.79, Recall 0.97   |
| **DME**    | Precision 0.87, but **recall plummets to 0.19** > many DME cases missed | Strong recovery with 0.95 precision and 0.69 recall  |
| **DRUSEN** | Only 0.33 recall > majority of DRUSEN missed                            | Much better at 0.81 recall and 0.97 precision        |
| **NORMAL** | Very high recall (0.99), decent precision (0.68)                        | Improved to precision 0.82 and perfect recall (1.00) |

##### Key Observations
- DME and DRUSEN were consistently the hardest to detect, especially in the loss-monitored model. These classes are known to have overlapping visual features in OCT images, which may explain the confusion.
- The loss-monitoring model overpredicted CNV, as shown by its low precision. This likely contributed to misclassification of DME/DRUSEN as CNV.
- The accuracy-monitored model shows balanced improvements across all classes, especially reducing false negatives on critical pathological conditions (DME and DRUSEN).

#### Clinical Relevance:
While the accuracy-monitored model performs much better, clinical use demands even higher recall and precision — particularly for pathological classes like DME and DRUSEN, where missing a case can delay treatment. Further improvements such as class balancing, transfer learning, or per-class tuning would be necessary before considering real-world deployment.


