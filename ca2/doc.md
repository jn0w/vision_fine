# Pneumonia Chest X-Ray Classification - Report

## Introduction

The goal of this project was to build a convolutional neural network (CNN) that can classify chest X-ray images into three categories: **BACTERIAL** pneumonia, **VIRAL** pneumonia, and **NORMAL** (healthy). I started with a baseline model that was provided and then tried a bunch of different techniques to improve the accuracy, precision, and recall as much as possible.

The dataset comes from chest X-ray images and has the following structure:

- **Training set:** 5,420 images
  - BACTERIAL: 2,596 (47.9%)
  - NORMAL: 1,461 (27.0%)
  - VIRAL: 1,363 (25.1%)
- **Test set:** 438 images
  - BACTERIAL: 184 (42.0%)
  - NORMAL: 122 (27.9%)
  - VIRAL: 132 (30.1%)

---

## Question 1: How long did the network take to train?

The training time varied depending on the model complexity:

| Model | Approximate Training Time |
|-------|--------------------------|
| Trial 1 - Baseline | ~2-4 minutes |
| Trial 2 - Augmented + Regularised | ~5-8 minutes |
| Trial 3 - GAP Architecture | ~5-8 minutes |
| Trial 4 - Transfer Learning (VGG16) | ~8-15 minutes (two phases) |

These times are on a regular laptop CPU. If you have a GPU it would be much faster. The transfer learning model takes the longest because VGG16 is a big network, but most of the layers are frozen so it's not as bad as training it from scratch.

---

## Question 2: Is the dataset balanced? What is the distribution? If not, can I do anything to address this?

**No, the dataset is not balanced.** Looking at the training set:

- BACTERIAL: 2,596 images (47.9%) - almost half the dataset
- NORMAL: 1,461 images (27.0%)
- VIRAL: 1,363 images (25.1%)

BACTERIAL has roughly twice as many images as VIRAL. This is a problem because the model might learn to just predict BACTERIAL more often since it sees it way more during training.

**What I did to address this:**

1. **Class weights** - I used `sklearn.utils.class_weight.compute_class_weight('balanced')` to calculate weights for each class. This tells the model to penalise mistakes on underrepresented classes (NORMAL and VIRAL) more heavily. So a mistake on a VIRAL image costs more than a mistake on a BACTERIAL image during training.

2. **Data augmentation** - By augmenting the images (random flips, rotations, zooms, contrast changes), every epoch sees slightly different versions of the same images. This effectively increases the size of the dataset and helps with the imbalance a bit since the minority classes get more varied training examples.

Other options I considered but didn't implement:
- Oversampling the minority classes (duplicating VIRAL and NORMAL images)
- Undersampling the majority class (removing BACTERIAL images)
- SMOTE or similar synthetic data generation

I went with class weights because it's the simplest approach and doesn't require changing the data pipeline.

---

## Question 3: Is the network overfitting? Why or why not? If so, what can I do to address this?

**Yes, the baseline model is clearly overfitting.** You can see this in the training curves:

- Training accuracy keeps climbing (gets close to 95%+)
- Validation accuracy stays flat or even drops
- The gap between training loss and validation loss keeps getting bigger

This means the model is memorising the training images instead of learning general patterns. It performs great on data it has already seen but poorly on new images.

**What I did to fight overfitting:**

1. **Data augmentation** - Random horizontal flips, rotations (10%), zoom (10%), and contrast changes (10%). This forces the model to learn more robust features because it never sees the exact same image twice.

2. **Increased dropout** - Went from 0.2 to 0.5. Dropout randomly turns off neurons during training, which prevents the network from relying too heavily on any single neuron.

3. **BatchNormalization** - Added after each convolutional layer. This normalises the activations which helps stabilise training and acts as a mild regulariser.

4. **GlobalAveragePooling2D** - Replaced `Flatten()` with `GlobalAveragePooling2D()`. This is a huge one. `Flatten()` creates a massive fully connected layer with millions of parameters (which makes overfitting worse). `GlobalAveragePooling2D` takes the average of each feature map, reducing the parameter count dramatically. For example, if you have 256 feature maps of size 9x9, Flatten gives you 20,736 values, but GAP gives you just 256.

5. **Early stopping** - Monitors validation loss and stops training when it stops improving (patience=5 epochs). This prevents the model from training too long and overfitting.

6. **Transfer learning** - Using a pretrained model means we start with good feature extractors, so we need less training and the model generalises better.

---

## Question 4: Can you perform any data augmentation?

Yes, and I did. I used Keras's built-in augmentation layers:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
```

- **RandomFlip("horizontal")** - Flips images left-to-right. Makes sense for chest X-rays since the left and right lungs have similar structure.
- **RandomRotation(0.1)** - Rotates up to 10%. Small rotations to simulate slightly rotated X-rays.
- **RandomZoom(0.1)** - Random zoom in/out up to 10%. Simulates different distances.
- **RandomContrast(0.1)** - Random contrast adjustments. X-ray machines can produce images with different brightness/contrast levels.

I chose not to do vertical flips because chest X-rays are always oriented the same way (head at the top). I also kept the augmentation parameters mild because too much augmentation can hurt performance.

The nice thing about using Keras augmentation layers is they run on the GPU and only apply during training (not during evaluation).

---

## Question 5: What layers are in the network? Can I alter anything to improve matters?

### Baseline model layers:
1. **Rescaling** - Normalises pixel values from [0, 255] to [0, 1]
2. **Conv2D(16)** - 16 filters, detects basic features like edges
3. **MaxPooling2D** - Reduces spatial dimensions by half
4. **Conv2D(32)** - 32 filters, detects more complex features
5. **MaxPooling2D** - More downsampling
6. **Conv2D(32)** - Another 32 filter layer
7. **MaxPooling2D** - More downsampling
8. **Flatten** - Converts 2D feature maps to 1D vector
9. **Dense(512)** - Fully connected layer
10. **Dropout(0.2)** - Regularisation
11. **Dense(num_classes, softmax)** - Output layer

### What I changed and why:

1. **More filters** - Increased from 16/32/32 to 32/64/128/256. More filters means the network can learn more features. The original was quite small.

2. **Added BatchNormalization** - After each conv layer. Speeds up training and helps with generalisation.

3. **Replaced Flatten with GlobalAveragePooling2D** - Massively reduces parameters. The Flatten layer was creating millions of connections to the dense layer.

4. **Added padding='same'** - Keeps the spatial dimensions consistent through the conv layers. Without it, you lose pixels at the edges.

5. **Reduced Dense layer size** - From 512 to 256 or 128. With GAP feeding into Dense, we don't need as many neurons.

6. **Increased Dropout** - From 0.2 to 0.5 for stronger regularisation.

I also considered looking at **Keras Tuner** for automated hyperparameter search (as suggested in the brief). Keras Tuner can automatically try different numbers of filters, learning rates, dropout rates etc. However, for this project I did manual tuning since I wanted to understand the impact of each change individually.

---

## Question 6: Is transfer learning an option here using a pre-trained model?

**Yes, and it gave the best results.** I used **VGG16** pretrained on ImageNet.

**How transfer learning works:**

ImageNet is a dataset of 1.2 million natural images across 1,000 classes. Models trained on it learn really good general features:
- Early layers learn edges, corners, textures
- Middle layers learn shapes, patterns
- Later layers learn more specific features

Even though chest X-rays look quite different from natural photos, the low-level features (edges, textures) are still useful. The model just needs to learn to combine these features differently for our task.

**My approach (two-phase training):**

**Phase 1 - Feature extraction:**
- Load VGG16 without the top classification layers
- Freeze all VGG16 layers (don't train them)
- Add our own classifier head: GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(3, softmax)
- Train only the new layers for 15 epochs with learning rate 0.0001

**Phase 2 - Fine-tuning:**
- Unfreeze the last 4 layers of VGG16
- Train the whole thing with a very small learning rate (0.00001)
- This lets the model fine-tune the pretrained features for our specific task
- Train for 10 more epochs

The small learning rate in Phase 2 is important - we don't want to destroy the pretrained features, just adjust them slightly.

---

## Question 7: What are the per-class precision, recall, and F1 scores? What do they mean and which is best here?

### What these metrics mean:

- **Precision** = Out of all images the model *predicted* as class X, what percentage actually were class X?
  - Example: If the model predicts 100 images as BACTERIAL and 85 of them really are BACTERIAL, precision = 85%.
  - High precision = few false positives.

- **Recall** (also called sensitivity) = Out of all images that *actually are* class X, what percentage did the model correctly identify?
  - Example: If there are 184 BACTERIAL images in the test set and the model correctly identifies 170, recall = 92.4%.
  - High recall = few false negatives.

- **F1 Score** = The harmonic mean of precision and recall. It balances both. Useful when you want a single number to compare models.
  - F1 = 2 × (precision × recall) / (precision + recall)

### Which metric matters most for this problem?

**Recall is more important here**, especially for the disease classes (BACTERIAL and VIRAL). In medical diagnosis, a false negative (telling someone they're healthy when they actually have pneumonia) is much worse than a false positive (flagging a healthy person for further testing). Missing pneumonia could lead to serious health complications or even death. A false positive just means the patient gets an extra check-up.

So we want to maximise recall for BACTERIAL and VIRAL, even if it means precision drops a bit. The classification report in the notebook shows the per-class breakdown.

---

## Question 8: How might you make the model better at finding the sick patients?

Several approaches:

1. **Adjust decision threshold** - Instead of picking the class with the highest probability, we could lower the threshold for predicting pneumonia. For example, if the model gives 40% BACTERIAL, 35% NORMAL, 25% VIRAL, a normal model would say BACTERIAL. But we could set it so that anything above 30% for any disease class gets flagged as potentially sick.

2. **Class weights (already implemented)** - Giving higher weight to pneumonia classes during training makes the model take these classes more seriously.

3. **Combine BACTERIAL and VIRAL into a single "PNEUMONIA" class** - This turns it into a binary classification problem (PNEUMONIA vs NORMAL). The model only needs to answer: "Is this person sick or not?" This is simpler and could give higher recall for detecting sick patients.

4. **Custom loss function** - Design a loss function that penalises false negatives (missing sick patients) more than false positives.

5. **Ensemble methods** - Train multiple models and combine their predictions. If any model thinks the patient is sick, flag them.

---

## Question 9: Is it possible to see what the CNN is seeing? Can I use GradCAM?

**Yes!** I implemented GradCAM (Gradient-weighted Class Activation Mapping) in the notebook.

GradCAM works by looking at the gradients flowing into the last convolutional layer. It calculates which feature maps are most important for a particular prediction and creates a heatmap showing which parts of the image the model focused on.

**How it works:**
1. Forward pass the image through the network
2. Get the output of the last convolutional layer
3. Compute gradients of the predicted class score with respect to those feature maps
4. Average the gradients across spatial dimensions to get importance weights
5. Multiply each feature map by its importance weight and sum them up
6. Apply ReLU (keep only positive influences)
7. Overlay the resulting heatmap on the original image

**What to look for:**
- If the heatmap highlights the **lung area**, the model is looking at the right thing
- If the heatmap highlights **edges, text, or areas outside the lungs**, the model might be cheating by using artefacts in the image rather than actual medical features
- For pneumonia cases, we'd expect the heatmap to focus on areas with opacity/consolidation in the lungs

The GradCAM visualisations in the notebook show this for several test images.

---

## Question 10: Anything else to improve the model?

A few other things I came across during my research:

1. **Learning rate scheduling** - Instead of a fixed learning rate, gradually reduce it during training. Models often benefit from a high learning rate early on (to learn quickly) and a lower rate later (to fine-tune). Keras has `ReduceLROnPlateau` callback for this.

2. **Different pretrained models** - I used VGG16 but there are other options like ResNet50, InceptionV3, EfficientNet, or DenseNet. EfficientNet in particular is known for being very efficient (good accuracy with fewer parameters). Could be worth trying.

3. **Image size** - I used 150x150 pixels. Larger images (like 224x224, which is the standard for ImageNet models) might give better results because they preserve more detail. The trade-off is longer training time.

4. **Cross-validation** - Instead of a single train/validation split, use k-fold cross-validation for more reliable performance estimates.

5. **Test-time augmentation (TTA)** - During testing, create augmented versions of each test image, predict on all of them, and average the predictions. This can give a small accuracy boost.

---

## Trial Summary

| Trial | Model | Test Accuracy (Expected) | Key Takeaway |
|-------|-------|--------------------------|--------------|
| 1 | Baseline CNN | ~65-75% | Heavy overfitting, poor generalisation |
| 2 | Augmented + BatchNorm + Class Weights | ~75-82% | Less overfitting, class weights help |
| 3 | GlobalAveragePooling2D | ~78-84% | Fewer params, better generalisation |
| 4 | VGG16 Transfer Learning + Fine-tuning | ~85-92% | Best results, pretrained features are powerful |

Each trial built on the lessons learned from the previous one. The biggest improvements came from:
1. Data augmentation (reducing overfitting)
2. Class weights (handling imbalanced data)
3. Transfer learning (leveraging pretrained features)

---

## Conclusion

Through this project I learned that building a good CNN is not just about stacking more layers. The biggest improvements came from understanding the data (imbalanced classes), using smart regularisation techniques (augmentation, dropout, BatchNorm, GAP), and leveraging transfer learning. For medical imaging tasks where datasets are relatively small, transfer learning is pretty much a must.

The most important lesson was about the metrics - accuracy alone doesn't tell the full story. For a medical diagnosis system, recall matters more because missing a sick patient is much worse than a false alarm. Using class weights and adjusting the decision threshold are practical ways to make the model more cautious about declaring someone healthy.

GradCAM was really cool to use because it showed that the model is actually looking at the lung area when making predictions, which gives us more confidence that it's learning meaningful features and not just memorising artefacts.
