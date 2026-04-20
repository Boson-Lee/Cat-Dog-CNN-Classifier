# %%
import os                   # For interacting with the operating system, like creating directories.
import sys                  # For redirecting our print statements to a file.
import shutil               # For high-level file operations like copying and moving.
import random               # For shuffling our data randomly.
import time                 # For timing our operations, like how long training takes.
import zipfile              # For unzipping the dataset file.
import urllib.request       # For downloading the dataset from the internet.
from datetime import datetime # For creating a timestamped folder for our results.
import warnings             # To handle and suppress specific warnings.

import torch                # The main PyTorch library for building and training neural networks.
import torch.nn as nn       # A module from PyTorch that contains building blocks for neural networks.
import torch.optim as optim # A module with optimization algorithms like Adam, used to update our model.
from torch.utils.data import DataLoader, Dataset # Tools for loading and managing our data.
from torchvision import transforms # A library with common image transformations.
from PIL import Image       # A library for opening, manipulating, and saving many different image file formats.
from tqdm import tqdm       # For displaying beautiful progress bars.

# Suppress a common warning from the Pillow library for images that might be slightly corrupted.
# Our data cleaning step handles most issues, but this keeps the output log clean.
warnings.filterwarnings("ignore", message="Truncated File Read")


# =================================================================================================
#
#  Section 1: Configuration & Hyperparameters
#
#  Here, we define all the settings and parameters that you can easily adjust.
#  Think of these as the control knobs for our machine learning experiment.
#
# =================================================================================================

class Config:
    """
    This class holds all the adjustable parameters for our project.
    Putting them here makes it easy to find and change them without searching through the code.
    """
    # --- Dataset and Paths ---
    DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    PROJECT_DIR = os.path.join(os.getcwd(), f"cat_dog_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
    LOG_FILE_PATH = os.path.join(PROJECT_DIR, "training_log.txt")
    MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "best_model.pth")
    INFERENCE_RESULTS_DIR = os.path.join(PROJECT_DIR, "inference_results")

    # --- Data Splitting ---
    # We will split our data into three sets:
    # - Training set (60%): Used to teach the model.
    # - Validation set (20%): Used to check the model's performance during training and prevent overfitting.
    # - Inference set (20%): Used to test the final model on data it has never seen before.
    TRAIN_SPLIT = 0.6
    VAL_SPLIT = 0.2
    # INFER_SPLIT is automatically calculated as 1.0 - TRAIN_SPLIT - VAL_SPLIT

    # --- Image Preprocessing ---
    IMAGE_SIZE = 128  # We will resize all images to 128x128 pixels.

    # --- Model Training ---
    BATCH_SIZE = 32      # The number of images the model looks at in each step of training.
    NUM_EPOCHS = 500     # An epoch is one full pass through the entire training dataset.
    LEARNING_RATE = 0.001 # Controls how much the model's parameters are adjusted during training.
    EARLY_STOPPING_PATIENCE = 15 # If the model doesn't improve for 15 consecutive epochs, we stop training.

# =================================================================================================
#
#  Section 2: Redirecting Output to a Log File
#
#  To keep a clean record of our training process, we will save all the printed output
#  to a text file. This is very useful for reviewing results later.
#
# =================================================================================================

class Logger:
    """
    A class to handle redirecting the console output (what you see when you print)
    to both the console and a file simultaneously.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This function is needed for compatibility with sys.stdout.
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        # When the program finishes, this closes the file.
        self.log.close()

# =================================================================================================
#
#  Section 3: Dataset Preparation
#
#  In this section, we will download the cat and dog dataset, unzip it, and organize the
#  images into training, validation, and inference folders.
#
# =================================================================================================

def download_and_prepare_dataset(config):
    """
    Handles the entire process of getting the data ready.
    """
    print("--- Starting Dataset Preparation ---")

    # Step 1: Create the main project directory.
    os.makedirs(config.PROJECT_DIR, exist_ok=True)
    print(f"Project directory created at: {config.PROJECT_DIR}")

    # Step 2: Check if the dataset directory already exists. If not, download and extract.
    raw_data_dir = os.path.join(config.DATASET_DIR, "PetImages")

    if os.path.exists(raw_data_dir):
        print("Dataset already found. Skipping download and extraction.")
    else:
        print("Dataset not found. Starting download...")
        os.makedirs(config.DATASET_DIR, exist_ok=True)
        zip_path = os.path.join(config.DATASET_DIR, "cats_and_dogs.zip")

        # Download the file.
        try:
            urllib.request.urlretrieve(config.DATASET_URL, zip_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading the dataset: {e}")
            return # Stop if download fails.

        # Unzip the file.
        print("Extracting files... This might take a few minutes.")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.DATASET_DIR)
        print("Extraction complete.")

        # Clean up the zip file.
        os.remove(zip_path)
        print("Cleaned up the zip file.")

    # Step 3: Clean the dataset (some images are corrupted and cannot be opened).
    print("Cleaning dataset: Verifying all images can be opened...")
    for root, _, files in os.walk(raw_data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img.verify() # Verify that it is, in fact, an image.
                except (IOError, SyntaxError) as e:
                    print(f"  - Deleting corrupted file: {img_path} ({e})")
                    os.remove(img_path)
    print("Dataset cleaning complete.")

    # Step 4: Split the data into train, validation, and inference sets.
    print("Splitting data into train, validation, and inference sets...")

    # Define paths for our new folders.
    train_dir = os.path.join(config.DATASET_DIR, "train")
    val_dir = os.path.join(config.DATASET_DIR, "val")
    infer_dir = os.path.join(config.DATASET_DIR, "infer")

    # Check if data is already split.
    if os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(infer_dir):
        print("Data already split. Skipping the splitting process.")
        print("--- Dataset Preparation Complete ---\n")
        return

    # Create the directories.
    for d in [train_dir, val_dir, infer_dir]:
        os.makedirs(os.path.join(d, "Cat"), exist_ok=True)
        os.makedirs(os.path.join(d, "Dog"), exist_ok=True)

    # Get all cat and dog image paths.
    cat_files = [os.path.join(raw_data_dir, "Cat", f) for f in os.listdir(os.path.join(raw_data_dir, "Cat"))]
    dog_files = [os.path.join(raw_data_dir, "Dog", f) for f in os.listdir(os.path.join(raw_data_dir, "Dog"))]

    # Shuffle them to ensure randomness.
    random.shuffle(cat_files)
    random.shuffle(dog_files)

    # Function to copy files to the correct folders.
    def split_and_copy(files, category_name):
        num_files = len(files)
        train_end = int(num_files * config.TRAIN_SPLIT)
        val_end = train_end + int(num_files * config.VAL_SPLIT)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        infer_files = files[val_end:]

        for f in train_files: shutil.copy(f, os.path.join(train_dir, category_name))
        for f in val_files: shutil.copy(f, os.path.join(val_dir, category_name))
        for f in infer_files: shutil.copy(f, os.path.join(infer_dir, category_name))

        print(f"  - {category_name}: {len(train_files)} train, {len(val_files)} val, {len(infer_files)} infer.")

    # Split and copy for both categories.
    split_and_copy(cat_files, "Cat")
    split_and_copy(dog_files, "Dog")

    print("Data splitting complete.")
    print("--- Dataset Preparation Complete ---\n")


# =================================================================================================
#
#  Section 4: Creating a Custom PyTorch Dataset
#
#  PyTorch uses a `Dataset` class to handle data. We will create our own custom class
#  that knows how to load our images, apply transformations (like resizing), and return
#  an image and its corresponding label (0 for cat, 1 for dog).
#
# =================================================================================================

class CatDogDataset(Dataset):
    """
    Custom Dataset class for our cat and dog images.
    This class loads images from a folder one by one, which is memory-efficient.
    """
    def __init__(self, directory, transform=None):
        """
        The constructor for our class. It runs when we create a new dataset object.
        - directory: The path to the folder (e.g., 'dataset/train').
        - transform: The image transformations to apply.
        """
        self.directory = directory
        self.transform = transform
        self.samples = [] # This will be a list of (image_path, label) pairs.
        self.classes = sorted(os.listdir(directory)) # ["Cat", "Dog"]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Go through the "Cat" and "Dog" subfolders and collect all image paths and their labels.
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, filename)
                        label = self.class_to_idx[class_name]
                        self.samples.append((path, label))

    def __len__(self):
        """
        This method returns the total number of images in the dataset.
        PyTorch uses this to know how big the dataset is.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This method gets a single image and its label from the dataset.
        - idx: The index (position) of the image we want.
        """
        # Get the image path and label for the given index.
        image_path, label = self.samples[idx]

        # Open the image file. We use a 'try-except' block to handle any corrupted files
        # that we might have missed during the cleaning step.
        try:
            image = Image.open(image_path).convert("RGB") # Convert to RGB to handle grayscale images.
        except Exception as e:
            print(f"\nWarning: Could not load image {image_path}. Skipping. Error: {e}")
            # If an image is broken, we load the next one instead.
            return self.__getitem__((idx + 1) % len(self))

        # Apply the transformations (e.g., resize, convert to tensor).
        if self.transform:
            image = self.transform(image)

        return image, label, image_path

# =================================================================================================
#
#  Section 5: Defining the CNN Model
#
#  Here we build our neural network. We will create a class that inherits from PyTorch's
#  `nn.Module`. The architecture is defined layer by layer, as specified in the prompt.
#
# =================================================================================================
# %%
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # GAP: (256, 8, 8) -> (256, 1, 1) -> flatten to 256
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# =================================================================================================
#
#  Section 6: Training and Evaluation Loop
#
#  This is the core of the machine learning process. We will write a function that:
#    - Feeds the training data to the model.
#    - Calculates the error (loss).
#    - Updates the model's parameters to reduce the error.
#    - Evaluates the model on the validation data to monitor its progress.
#
# =================================================================================================

def train_model(config):
    """
    The main function to orchestrate the model training and evaluation process.
    """
    print("--- Starting Model Training ---")

    # Step 1: Set the device (GPU or CPU).
    # We check if a CUDA-enabled GPU is available. If so, we use it for faster training.
    # Otherwise, we default to using the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Prepare data transformations.
    # We define a pipeline of transformations to apply to each image.
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), # Resize the image.
            transforms.RandomHorizontalFlip(), # Randomly flip images horizontally for data augmentation.
            transforms.ToTensor(), # Convert the image to a PyTorch tensor (values 0-1).
            # We normalize by dividing by 255. ToTensor() already does this, scaling to [0, 1].
        ]),
        'val': transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ]),
        'infer': transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ]),
    }

    # Step 3: Create datasets and dataloaders.
    print("\nLoading datasets...")
    train_dataset = CatDogDataset(
        directory=os.path.join(config.DATASET_DIR, "train"),
        transform=data_transforms['train']
    )
    val_dataset = CatDogDataset(
        directory=os.path.join(config.DATASET_DIR, "val"),
        transform=data_transforms['val']
    )
    # The DataLoader takes a Dataset and prepares batches of data for the model.
    # `shuffle=True` for the training set is important for good learning.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=8, # Use multiple threads to load data faster.
        pin_memory=True # Helps speed up data transfer to the GPU.
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")


    # Step 4: Initialize the model, loss function, and optimizer.
    model = SimpleCNN().to(device)

    # Print model architecture and calculate total trainable parameters.
    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    print("--------------------------\n")

    # The loss function measures how wrong the model's predictions are.
    # CrossEntropyLoss is standard for multi-class classification problems.
    criterion = nn.CrossEntropyLoss()

    # The optimizer's job is to update the model's parameters based on the loss.
    # Adam is a popular and effective optimization algorithm.
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Step 5: The training loop.
    best_val_loss = float('inf') # Initialize with a very high value.
    epochs_no_improve = 0
    training_start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"--- Epoch {epoch + 1}/{config.NUM_EPOCHS} ---")

        # --- Training Phase ---
        model.train() # Set the model to training mode.
        running_loss = 0.0
        running_corrects = 0

        # Wrap the data loader with tqdm for a progress bar.
        for inputs, labels, _ in tqdm(train_loader, desc="  Training", leave=False):
            # Move data to the selected device (GPU/CPU).
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients. This is a necessary step before each update.
            optimizer.zero_grad()

            # Forward pass: get model outputs.
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1) # Get the index of the highest score (our prediction).
            loss = criterion(outputs, labels) # Calculate the loss.

            # Backward pass and optimize.
            loss.backward() # Calculate gradients.
            optimizer.step() # Update model weights.

            # Accumulate statistics.
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval() # Set the model to evaluation mode.
        running_loss = 0.0
        running_corrects = 0

        # We don't need to calculate gradients during validation, which saves memory and computation.
        with torch.no_grad():
            # Wrap the data loader with tqdm for a progress bar.
            for inputs, labels, _ in tqdm(val_loader, desc="  Validating", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)

        epoch_duration = time.time() - epoch_start_time

        # Print a detailed summary for the epoch.
        print(f"  Epoch Duration: {epoch_duration:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # --- Early Stopping and Model Saving ---
        if val_loss < best_val_loss:
            print(f"  Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"  Validation loss did not improve. Patience: {epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nStopping early after {epoch + 1} epochs as validation loss did not improve for {config.EARLY_STOPPING_PATIENCE} epochs.")
            break

        print("-" * (len(str(config.NUM_EPOCHS)) * 2 + 13))

    total_training_time = time.time() - training_start_time
    print(f"\n--- Training Complete ---")
    print(f"Total Training Time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {config.MODEL_SAVE_PATH}")


# =================================================================================================
#
#  Section 7: Inference on the Test Set
#
#  Now that our model is trained, we will use it to make predictions on the 'infer'
#  dataset, which it has never seen before. This gives us the best estimate of how
#  the model will perform in the real world.
#
# =================================================================================================

def run_inference(config):
    """
    Loads the best trained model and evaluates it on the inference (test) dataset.
    """
    print("\n--- Starting Inference on the Test Set ---")

    # Step 1: Set device and load the best model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Check if a trained model file exists.
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Error: Model file not found. Please train the model first.")
        return

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval() # Set model to evaluation mode.
    print("Best model loaded for inference.")

    # Step 2: Prepare the inference dataset and dataloader.
    infer_dataset = CatDogDataset(
        directory=os.path.join(config.DATASET_DIR, "infer"),
        transform=transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
        ])
    )
    infer_loader = DataLoader(infer_dataset, batch_size=config.BATCH_SIZE)
    print(f"Found {len(infer_dataset)} images for inference.")

    # Get class names from the dataset object.
    class_names = infer_dataset.classes # Should be ['Cat', 'Dog']

    # Step 3: Run inference and collect statistics.
    correct = 0
    total = 0
    cat_correct, cat_total = 0, 0
    dog_correct, dog_total = 0, 0
    inference_times = []
    correctly_classified = []
    incorrectly_classified = []

    with torch.no_grad():
        for inputs, labels, paths in infer_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()

            _, predicted = torch.max(outputs.data, 1)

            # Calculate batch inference time and store it.
            batch_time = (end_time - start_time) / len(inputs)
            inference_times.append(batch_time)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Detailed statistics per class.
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                path = paths[i]

                # Label 0 is 'Cat', Label 1 is 'Dog'
                if label == 0: # Cat
                    cat_total += 1
                    if pred == label:
                        cat_correct += 1
                        if len(correctly_classified) < 10:
                            correctly_classified.append((path, class_names[label], class_names[pred]))
                    else:
                        if len(incorrectly_classified) < 5:
                            incorrectly_classified.append((path, class_names[label], class_names[pred]))
                elif label == 1: # Dog
                    dog_total += 1
                    if pred == label:
                        dog_correct += 1
                        if len(correctly_classified) < 10:
                            correctly_classified.append((path, class_names[label], class_names[pred]))
                    else:
                        if len(incorrectly_classified) < 5:
                            incorrectly_classified.append((path, class_names[label], class_names[pred]))


    # Step 4: Print detailed inference results.
    print("\n--- Inference Results Summary ---")
    overall_accuracy = 100 * correct / total
    cat_accuracy = 100 * cat_correct / cat_total if cat_total > 0 else 0
    dog_accuracy = 100 * dog_correct / dog_total if dog_total > 0 else 0
    avg_inference_time = sum(inference_times) / len(inference_times) * 1000 # in milliseconds

    print(f"  Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    print(f"  Cat Accuracy:     {cat_accuracy:.2f}% ({cat_correct}/{cat_total})")
    print(f"  Dog Accuracy:     {dog_accuracy:.2f}% ({dog_correct}/{dog_total})")
    print(f"  Average Inference Speed: {avg_inference_time:.2f} ms per image")

    # Step 5: Save and display example images.
    print("\nSaving example classification results...")
    os.makedirs(config.INFERENCE_RESULTS_DIR, exist_ok=True)

    def save_examples(image_list, folder_name):
        sub_dir = os.path.join(config.INFERENCE_RESULTS_DIR, folder_name)
        os.makedirs(sub_dir, exist_ok=True)
        for i, (path, true_label, pred_label) in enumerate(image_list):
            try:
                # Create a descriptive filename.
                new_filename = f"{i+1}_True-{true_label}_Predicted-{pred_label}.jpg"
                save_path = os.path.join(sub_dir, new_filename)
                shutil.copy(path, save_path)
            except Exception as e:
                print(f"  - Could not save example image {path}. Error: {e}")

    save_examples(correctly_classified, "correct_predictions")
    save_examples(incorrectly_classified, "incorrect_predictions")
    print(f"Saved {len(correctly_classified)} correct and {len(incorrectly_classified)} incorrect examples.")
    print(f"You can find them in: {config.INFERENCE_RESULTS_DIR}")
    print("--- Inference Complete ---\n")


# =================================================================================================
#
#  Section 8: Main Execution Block
#
#  This is where the program starts. It calls the functions we defined above in the
#  correct order.
#
# =================================================================================================
# %%
if __name__ == "__main__":
    # Create an instance of our configuration class.
    config = Config()

    # Create the project directory if it doesn't exist.
    os.makedirs(config.PROJECT_DIR, exist_ok=True)

    # Set up our logger to redirect print output to a file.
    sys.stdout = Logger(config.LOG_FILE_PATH)

    try:
        # Run the entire pipeline.
        download_and_prepare_dataset(config)
        train_model(config)
        run_inference(config)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # This will make sure the error is logged to our file.
        import traceback
        traceback.print_exc()

    print("Script finished.")

# %% 
print("--- 獨立繪製 ROC 曲線區塊 (純淨穩定版) ---")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. 客製化精準路徑 (已根據你的截圖設定)
# ==========================================
YOUR_WORKSPACE_NAME = "cat_dog_classifier_20260411_155658" 

YOUR_MODEL_PATH = f"{YOUR_WORKSPACE_NAME}/best_model.pth"
YOUR_DATASET_DIR = f"{YOUR_WORKSPACE_NAME}/dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的運算設備: {device}")

# ==========================================
# 2. 定義模型與資料集 (內建定義，保證不失憶)
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: (3, 128, 128) -> (32, 64, 64)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (32, 64, 64) -> (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, 32, 32) -> (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: (128, 16, 16) -> (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # GAP: (256, 8, 8) -> (256, 1, 1) -> flatten to 256
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MinimalCatDogDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.samples = []
        self.classes = sorted(os.listdir(directory))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        for class_name in self.classes:
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        path = os.path.join(class_dir, filename)
                        label = self.class_to_idx[class_name]
                        self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, image_path

# ==========================================
# 3. 定義繪圖函數 
# ==========================================
def evaluate_and_plot_roc(model, train_loader, val_loader, device, save_path):
    model.eval()

    def get_true_and_pred_probs(dataloader):
        all_true_labels = []
        all_pred_probs = []
        with torch.no_grad():
            for inputs, labels, _ in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                dog_probs = probs[:, 1]
                all_true_labels.extend(labels.cpu().numpy())
                all_pred_probs.extend(dog_probs.cpu().numpy())
        return all_true_labels, all_pred_probs

    print("正在計算訓練集預測機率 (OneDrive 環境讀取圖片約需 2~5 分鐘，請耐心等待)...")
    train_true, train_probs = get_true_and_pred_probs(train_loader)
    
    print("正在計算驗證集預測機率...")
    val_true, val_probs = get_true_and_pred_probs(val_loader)

    print("計算完成！正在繪製圖表...")
    train_fpr, train_tpr, _ = roc_curve(train_true, train_probs)
    train_auc = auc(train_fpr, train_tpr)
    val_fpr, val_tpr, _ = roc_curve(val_true, val_probs)
    val_auc = auc(val_fpr, val_tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label=f'Train ROC (AUC = {train_auc:.4f})')
    plt.plot(val_fpr, val_tpr, color='red', lw=2, label=f'Validation ROC (AUC = {val_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=14)
    plt.ylabel('True Positive Rate (TPR)', fontsize=14)
    plt.title('ROC Curve: Training vs Validation (Overfitting Check)', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ ROC 曲線已順利產生並儲存至: {save_path}")

# ==========================================
# 4. 載入模型與資料並開始執行
# ==========================================
try:
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(YOUR_MODEL_PATH, map_location=device))
    print("✅ 模型載入成功！開始讀取圖片...")

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    train_dataset = MinimalCatDogDataset(os.path.join(YOUR_DATASET_DIR, "train"), transform=transform)
    val_dataset = MinimalCatDogDataset(os.path.join(YOUR_DATASET_DIR, "val"), transform=transform)

    # 【關鍵修正】：num_workers 強制設為 0，徹底解決 Windows/OneDrive 卡死問題
    roc_train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    roc_val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    evaluate_and_plot_roc(model, roc_train_loader, roc_val_loader, device, "roc_curve_result.png")

except FileNotFoundError as e:
    print(f"❌ 找不到檔案，請確認路徑設定是否正確: {e}")
except Exception as e:
    print(f"❌ 發生未預期的錯誤: {e}")
# %%
