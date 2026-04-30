import os
import json

import torch
from PIL import Image
from torchvision import transforms

from model import resnet18

def main():
    # Determine the device to run inference on (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data preprocessing pipeline for the input image
    # Note: These normalization values are standard for ImageNet pre-trained models
    data_transform = transforms.Compose(
        [transforms.Resize(256),  # Resize the shorter edge to 256
         transforms.CenterCrop(224),  # Center crop a 224x224 patch
         transforms.ToTensor(),  # Convert image to PyTorch tensor [C, H, W]
         transforms.Normalize([0.485, 0.456, 0.406],  # Normalize with ImageNet mean
                              [0.229, 0.224, 0.225])])  # Normalize with ImageNet std

    # Load the input image
    img_path = "sunflower.png"         # TODO: Update this path to point to the image you want to classify
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    # Apply the transformation pipeline
    img = data_transform(img)

    # Expand the batch dimension to match model input requirements
    # Changes shape from [C, H, W] to [1, C, H, W] (e.g., [1, 3, 224, 224])
    img = torch.unsqueeze(img, dim=0)

    # Read the class dictionary mapping from the JSON file generated during training
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Initialize the ResNet-18 model
    # Ensure num_classes matches the number of classes you trained on
    model = resnet18(num_classes=5).to(device)

    # Load the trained model weights
    weights_path = "./resnet18_best_checkpoint.pth"         # TODO: Ensure this path points to your trained model weights
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Set the model to evaluation mode (disables Dropout, freezes BatchNorm)
    model.eval()

    # Disable gradient calculation for inference to save memory and speed up computation
    with torch.no_grad():
        # Perform inference and move the output back to CPU
        output = torch.squeeze(model(img.to(device))).cpu()

        # Apply softmax to convert raw logits into probabilities (sum to 1)
        predict = torch.softmax(output, dim=0)

        # Get the index of the class with the highest probability
        predict_cla = torch.argmax(predict).numpy()

    # Print the final predicted class and its probability
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    print(print_res)
    print("-" * 30)

    # Print the probabilities for all classes
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))

if __name__ == '__main__':
    main()