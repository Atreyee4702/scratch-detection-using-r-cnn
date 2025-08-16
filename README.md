
---

# Scratch Detection from Scratch

### Overview

This project does the task of detecting scratches on images containing text. The dataset consists of two categories:
1. **Good Images**: Images with clear, unobstructed text.
2. **Bad Images**: Images where text is obscured by scratches.

The model classifies images into these categories and generates a bounding box or mask for scratches on bad images.

Read the full report here : [https://github.com/sdjbabin/Scratch-Detection-from-scratch/blob/master/mowito_assignment_soham_chatterjee.pdf]

## Architecture of CNN model
![image](https://github.com/user-attachments/assets/20e88a1c-284e-4bb9-b54b-d2fd5b705587)

## Classification report

![image](https://github.com/user-attachments/assets/abb2fd06-cca7-4484-aace-c198098daaf4)

## Trying with U Net

![image](https://github.com/user-attachments/assets/feeb0efd-4203-448f-8203-3f036c32dcc5)

## Final masks prediction on bad images using Mask R CNN

![image](https://github.com/user-attachments/assets/303dcb2d-1d15-47b2-bd33-b3035b091be7)

![image](https://github.com/user-attachments/assets/3aa56e78-ea6a-4bd4-908b-2f83d030d16d)
![image](https://github.com/user-attachments/assets/8e3a7a68-5679-4722-8b80-de8b331beabc)
![image](https://github.com/user-attachments/assets/55999869-efab-4f8d-af4f-1bfc372cf2b9)




---

### Objectives

1. **Image Classification**: Classify images as good (no scratches) or bad (scratches present).
2. **Scratch Localization**: Generate bounding boxes or masks around scratches in bad images.
3. **User-defined Thresholds**: Allow users to set thresholds for scratch size to define a bad image.
4. **Generalization**: Expand the model to detect scratches on other surfaces, such as metallic objects and phone screens.

---

### Approach

#### 1. Dataset Preparation
The dataset was organized into:
- `train` and `train_masks` for training data.
- `test` and `test_masks` for testing data.

Bad image masks were converted into JSON annotations for instance segmentation tasks.

#### 2. Models Explored
- **U-Net**: Implemented for semantic segmentation but showed poor results due to inconsistent mask predictions under varying conditions.
- **Mask R-CNN**: Utilized for instance segmentation, fine-tuned for scratch detection. This model produced satisfactory results with accurate predictions on unseen data.

---

### Architecture and Training

#### 1. **Custom CNN for Classification**
Key parameters:
- Input size: 100x100
- Layers: 2 convolutional layers, 2 max-pooling layers, 1 fully connected layer.
- Optimizer: Adam
- Loss: Binary cross-entropy
- Metrics: Accuracy, recall for bad images.

#### 2. **Mask R-CNN for Instance Segmentation**
- Backbone: ResNet-50.
- Hyperparameters:
  - Learning rate: 0.005
  - Batch size: 2
  - Epochs: 3
- Results: Achieved better mask accuracy, especially on unseen images.

#### Training Code
```python
dataset = ScratchDataset(
    json_file='/content/annotations.json',
    img_dir='/content/images/UNet-PyTorch/data/train'
)

data_loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)

model = get_model_instance_segmentation(num_classes=2)
model.to(device)

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)

for epoch in range(3):
    loss = train_one_epoch(model, optimizer, data_loader, device)
    print(f"Epoch: {epoch}, Loss: {loss}")
```

---

### Results
- **U-Net**: Inconsistent mask predictions due to lighting variations.
- **Mask R-CNN**: Effective and accurate segmentation with confidence scores reflecting detection certainty.

---

### Usage

#### Prerequisites
- Python 3.7+
- PyTorch
- torchvision
- OpenCV
- PIL

#### Training
1. Download the dataset: [https://drive.google.com/file/d/1fICPduR2TPT6AulTPrugBCNhOo0qcC62/view?usp=sharing]
2. Run the notebooks

#### Pre-trained Model
Download the pre-trained model: [https://drive.google.com/drive/folders/1JWl_R2YZPlX4x1bdEtKGyAZBn-h2TsrF?usp=drive_link]

---

### General Remarks
1. Batch normalization improved metrics for bad images but lowered metrics for good images due to dampening of simpler features.
2. Larger batch sizes reduced validation loss but did not significantly alter performance.
3. Synthetic data (e.g., adding random scratches to good images) can further enhance bad image datasets.

---

### Future Work
1. Extend the model to detect scratches on other surfaces (e.g., phone screens, metallic objects).
2. Implement more advanced augmentation and synthetic data generation techniques.

---

### References
- [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)
- [PyTorch Instance Segmentation](https://pytorch.org/vision/stable/models.html)

--- 

You can edit the placeholders for dataset and model links. This README is structured to provide clarity and guide users in replicating your project.
