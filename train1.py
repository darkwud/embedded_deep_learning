import os
import sys
import json
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import torch.utils.data as data
import argparse

from model import resnet18
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    # Determine the device to run the model on (GPU if available, else CPU)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Create a directory to save model weights if it does not exist
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    # Read and split the dataset into training and validation sets
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path,
                                                                                               val_rate=args.val_rate)

    # Define data preprocessing and augmentation pipelines
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize with mean and std
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),  # Resize shorter edge to 256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    # Instantiate the custom training dataset
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # Instantiate the custom validation dataset
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    # Extract class names from the dataset directory structure
    class_list = sorted(os.listdir(args.data_path))
    class_num = len(class_list)
    vals = list(range(class_num))
    data_list = zip(class_list, vals)

    # Create a dictionary mapping class indices to class names
    cla_dict = dict((val, key) for key, val in data_list)
    print("Class mapping:", cla_dict)

    # Save the class dictionary to a JSON file for future inference
    json_str = json.dumps(cla_dict, indent=4, ensure_ascii=False)
    with open('class_indices.json', 'w', encoding='utf-8') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    # Calculate the optimal number of workers for data loading based on CPU cores
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # Configure the training data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    # Configure the validation data loader
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    val_num = len(val_loader)
    print("Validation batches:", val_num)

    # Initialize the ResNet-18 model
    net = resnet18()

    # Load pre-trained weights for transfer learning
    model_weight_path = args.weights
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # Optional: Freeze network layers to only train the final classification layer
    # for param in net.parameters():
    #     param.requires_grad = False

    # Modify the final fully connected (fc) layer to match the new dataset's class number
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, class_num)
    net.to(device)

    # Define the loss function (Cross Entropy for multi-class classification)
    loss_function = nn.CrossEntropyLoss()

    # Construct the optimizer (Adam), passing only the parameters that require gradients
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    # CSV file log
    epochs = args.epochs
    best_acc = 0.0
    save_path = './weights/resnet18_best.pth'
    train_steps = len(train_loader)

    # Create a CSV file to log loss and accuracy per epoch
    log_filename = f"loss_log_lr_{args.lr}.csv"
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_accuracy'])

    # Start the training loop over multiple epochs
    for epoch in range(epochs):
        # Set the model to training mode (enables Dropout and BatchNorm tracking)
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data

            # Step 1: Clear previous gradients
            optimizer.zero_grad()

            # Step 2: Forward pass
            logits = net(images.to(device))

            # Step 3: Compute the loss
            loss = loss_function(logits, labels.to(device))

            # Step 4: Backward pass to compute gradients
            loss.backward()

            # Step 5: Update model weights
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # Validation phase
        # Set the model to evaluation mode (disables Dropout, uses population stats for BatchNorm)
        net.eval()
        acc = 0.0

        # Disable gradient calculation to save memory and computation during inference
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))

                # Get the predicted class indices (highest probability)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        # Calculate average validation accuracy
        val_accurate = acc / (val_num * batch_size)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # Log the training loss and validation accuracy to the CSV file
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, running_loss / train_steps, val_accurate])

        # Save the model weights if the current accuracy is the best seen so far
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    # Parse command line arguments for hyperparameter tuning and configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val_rate', type=float, default=0.2, help='proportion of data used for validation')

    parser.add_argument('--weights', type=str, default='./resnet18-5c106cde.pth',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--data-path', type=str, default="./data/flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')
    opt = parser.parse_args()

    main(opt)