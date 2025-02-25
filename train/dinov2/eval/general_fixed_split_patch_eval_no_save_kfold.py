import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap
import wandb
from models.return_model import get_models, get_transforms
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import CustomImageDataset
import matplotlib.patches as mpatches
#import kfolds
from sklearn.model_selection import StratifiedKFold

from torchvision import transforms
#import rearrange
from einops import rearrange
parser = argparse.ArgumentParser(description="Feature extraction")
os.environ["WANDB__SERVICE_WAIT"] = "300"


parser.add_argument(
    "--model_name",
    help="name of model",
    default="dinov2_vits14",
    type=str,
)

parser.add_argument(
    "--experiment_name",
    help="name of experiment",
    default="",
    type=str,
)

parser.add_argument(
    "--num_workers",
    help="num workers to load data",
    default=4,
    type=int,
)

parser.add_argument(
    "--batch_size",
    help="num workers to load data",
    default=128,
    type=int,
)

parser.add_argument(
    "--image_path_train",
    help="path to csv file",
    default="",
    type=str,
)

parser.add_argument(
    "--image_path_test",
    help="path to csv file",
    default="",
    type=str,
)

parser.add_argument(
    "--run_path",
    "--model_path",
    help="path to run directory with models inside",
    default="",
    type=str,
)

parser.add_argument(
    "--run_name",
    help="name of wandb run",
    default="debug",
    type=str,
)

parser.add_argument(
    "--project_name",
    help="name of wandb project",
    default="debug",
    type=str,
)
parser.add_argument(
    "--knn",
    help="perform knn or not",
    default=True,
    type=bool,
)

parser.add_argument(
    "--evaluate_untrained_baseline",
    help="Set to true if original dino should be tested.",
    action="store_true",
)

parser.add_argument(
    "--logistic_regression",
    "--logistic-regression",
    "-log",
    help="perform logistic regression or not",
    default=True,
    type=bool,
)

parser.add_argument(
    "--umap",
    help="perform umap or not",
    default=True,
    type=bool,
)

#argument for single file or all in one
parser.add_argument(
    "--all_in_one",
    help="save all features in one file",
    default=False,
    type=bool,
)
#point size in umap
parser.add_argument(
    "--point_size",
    help="point size in umap",
    default=10.2,
    type=float,
)

parser.add_argument(
    "--umapv2",
    help="use the second version of umap if activated",
    default=True,
    type=bool,
)

#cross validation parameter
parser.add_argument(
    "--cross_val",
    help="perform cross validation",
    default=True,
    type=bool,
)



def extract_features(feature_extractor, dataloader, dataset, img_size_patch=224):



    print("extracting features..")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_features = []
    full_labels = []
    with torch.no_grad():
        feature_extractor.eval()

        for images, labels, names in tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)
            labels_np = labels.numpy()
            full_features.append(batch_features)
            full_labels.append(labels_np)

    full_features = torch.cat(full_features, dim=0)
    full_labels = np.concatenate(full_labels)

    return full_features, full_labels
#now extract but the features are transformed to numpy and not tensor concatenated
def extract_features_numpy(feature_extractor, dataloader, dataset):
    print("extracting features..")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_features = []
    full_labels = []
    with torch.no_grad():
        feature_extractor.eval()

        for images, labels, names in tqdm(dataloader):
            images = images.to(device)
            batch_features = feature_extractor(images)
            labels_np = labels.numpy()
            batch_features = batch_features.cpu().numpy()
            full_features.append(batch_features)
            full_labels.append(labels_np)

    full_features = np.concatenate(full_features, axis=0)
    full_labels = np.concatenate(full_labels)

    return full_features, full_labels
         





def sort_key(path):
    # Extract the numeric part from the directory name
    # Assuming the format is always like '.../train_xxxx/...'
    number_part = int(path.parts[-2].split("_")[1])
    return number_part


def main(args):
    image_paths = args.image_path_train
    image_test_paths = args.image_path_test
    model_name = args.model_name
    

    df = pd.read_csv(image_paths)
    df_test = pd.read_csv(image_test_paths)


    transform = get_transforms(model_name)

    # make sure encoding is always the same

    train_dataset = CustomImageDataset(df, transform=transform)
    test_dataset = CustomImageDataset(df_test, transform=transform)



    # Create data loaders for the  datasets
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # If you want to log the results with Weights & Biases (wandb), you can initialize a wandb run:
    wandb.init(project=args.project_name, name=args.experiment_name, config=args)

    if model_name in ["owkin", "resnet50", "resnet50_full", "remedis", "imagebind"]:
        sorted_paths = [None]
    elif model_name in ["dino_org"]:
        sorted_paths = list(Path(args.run_path).rglob("*.pth"))
    elif model_name in ["retccl", "ctranspath"]:
        sorted_paths = [Path(args.run_path)]
    elif model_name in ["dino_s_org", "dino_b_org", "dino_l_org", "dino_g_org"]:
        sorted_paths = [None]
    elif model_name in ["dinov2_s_org", "dinov2_b_org", "dinov2_l_org", "dinov2_g_org"]:
        sorted_paths = [None]
    elif model_name in ["dino_fm_b"]:
        sorted_paths = [Path(args.run_path)]
    else:
        sorted_paths = list(Path(args.run_path).rglob("*teacher_checkpoint.pth"))

    if len(sorted_paths) > 1:
        sorted_paths = sorted(sorted_paths, key=sort_key)
    if args.evaluate_untrained_baseline:
        sorted_paths.insert(0, None)

    for checkpoint in sorted_paths:
        if checkpoint is not None:
            parent_dir = checkpoint.parent
        else:
            parent_dir = Path(args.run_path) / (model_name + "_baseline")

        print("loading checkpoint: ", checkpoint)
        feature_extractor = get_models(model_name, saved_model_path=checkpoint)
        feature_dir = parent_dir / args.run_name

        train_dir = os.path.join(feature_dir, "train_data")
        test_dir = os.path.join(feature_dir, "test_data")

        if "_g_" in model_name:
            print("extracting features using batch numpy instead of tensor")
            train_data, train_labels = extract_features_numpy(feature_extractor, train_dataloader, train_dataset)
            test_data, test_labels = extract_features_numpy(feature_extractor, test_dataloader, test_dataset)
        else:
            print("extracting features using tensor")
            # extract features without saving them
            train_data, train_labels = extract_features(feature_extractor, train_dataloader, train_dataset)
            test_data, test_labels = extract_features(feature_extractor, test_dataloader, test_dataset)
            # to cpu and numpy
            train_data = train_data.cpu().numpy()
            test_data = test_data.cpu().numpy()
        print("features extracted")
        
        N_FOLDS = 5
        #create a datafreame with the following columns
        columns = ["accuracy", "balanced_accuracy", "weighted_f1", "algorithm", "fold"]
        df_tot = pd.DataFrame(columns=columns)
        #perform cross validation if activated then merge the features
        if args.cross_val:
            data = np.concatenate((train_data, test_data), axis=0)
            labels = np.concatenate((train_labels, test_labels), axis=0)
            skf = StratifiedKFold(n_splits=N_FOLDS)
    
            for fold, (train_index, test_index) in enumerate(skf.split(data, labels)):
                print(f"Fold {fold}")
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                #perform logistic regression
                if args.logistic_regression:
                    logreg_dir = parent_dir / "log_reg_eval"
                    loss, accuracy, balanced_acc, weighted_f1, report = train_and_evaluate_logistic_regression(X_train, y_train, X_test, y_test, logreg_dir, max_iter=1000)
                    #add the metrics to the dataframe
                    row = [accuracy, balanced_acc, weighted_f1, "logistic_regression", fold]
                    #add the row to the dataframe
                    df = pd.DataFrame([row], columns=columns )
                    df_tot = pd.concat([df_tot, df])
                #perform knn
                if args.knn:
                    knn_dir = parent_dir / "knn_eval"
                    knn_metrics, accuracy_1, balanced_acc_1, weighted_f1_1, accuracy_20, balanced_acc_20, weighted_f1_20 = perform_knn(X_train, y_train, X_test, y_test, knn_dir)
                    #add the metrics to the dataframe
                    row_1 = [accuracy_1, balanced_acc_1, weighted_f1_1, "knn_1", fold]
                    row_20 = [accuracy_20, balanced_acc_20, weighted_f1_20, "knn_20", fold]
                    #add the rows to the dataframe
                    df = pd.DataFrame([row_1, row_20], columns=columns )
                    df_tot = pd.concat([df_tot, df])

            df = df_tot              


            #get the mean of the metrics
            mean_metrics = df.groupby("algorithm").mean()
            print("mean metrics")
            print(mean_metrics)
            #get the std of the metrics
            std_metrics = df.groupby("algorithm").std()
            print("std metrics")
            print(std_metrics)

            #to wandb
            values = {
                "mean:": {
                "log_reg":{
                    "accuracy": mean_metrics.loc["logistic_regression"]["accuracy"],
                    "balanced_accuracy": mean_metrics.loc["logistic_regression"]["balanced_accuracy"],
                    "weighted_f1": mean_metrics.loc["logistic_regression"]["weighted_f1"]
                },
                "knn_1":{
                    "accuracy": mean_metrics.loc["knn_1"]["accuracy"],
                    "balanced_accuracy": mean_metrics.loc["knn_1"]["balanced_accuracy"],
                    "weighted_f1": mean_metrics.loc["knn_1"]["weighted_f1"]
                },
                "knn_20":{
                    "accuracy": mean_metrics.loc["knn_20"]["accuracy"],
                    "balanced_accuracy": mean_metrics.loc["knn_20"]["balanced_accuracy"],
                    "weighted_f1": mean_metrics.loc["knn_20"]["weighted_f1"]
                }
            },
            "std": {
                "log_reg":{
                    "accuracy": std_metrics.loc["logistic_regression"]["accuracy"],
                    "balanced_accuracy": std_metrics.loc["logistic_regression"]["balanced_accuracy"],
                    "weighted_f1": std_metrics.loc["logistic_regression"]["weighted_f1"]
                },
                "knn_1":{
                    "accuracy": std_metrics.loc["knn_1"]["accuracy"],
                    "balanced_accuracy": std_metrics.loc["knn_1"]["balanced_accuracy"],
                    "weighted_f1": std_metrics.loc["knn_1"]["weighted_f1"]
                },
                "knn_20":{
                    "accuracy": std_metrics.loc["knn_20"]["accuracy"],
                    "balanced_accuracy": std_metrics.loc["knn_20"]["balanced_accuracy"],
                    "weighted_f1": std_metrics.loc["knn_20"]["weighted_f1"]
                }

            }
            }
            wandb.log(values)
            


            
    

                    
            
            

            #plot the umap
            if args.umap:
                umap_dir = parent_dir / "umaps"
                if args.umapv2:
                    print("using umap v2")
                    umap_total = create_umap_v2(data, labels, umap_dir, train_dataset.class_to_label)
                else:
                    print("using umap v1")
                    umap_total = create_umap(data, labels, umap_dir, train_dataset.class_to_label)
                print("umap done")
                wandb.log({"umap_total": wandb.Image(umap_total)})
            




def process_file(file_name):
    with h5py.File(file_name, "r") as hf:
        features = torch.tensor(hf["features"][:]).tolist()
        label = int(hf["labels"][()])
    return features, label


def get_data(train_dir, test_dir):
    # Define the directories for train, validation, and test data and labels

    # Load training data into dictionaries
    train_features, train_labels = [], []
    test_features, test_labels = [], []

    train_files = list(Path(train_dir).glob("*.h5"))
    test_files = list(Path(test_dir).glob("*.h5"))

    with ThreadPoolExecutor() as executor:
        futures_train = [executor.submit(process_file, file_name) for file_name in train_files]

        for i, future_train in tqdm(enumerate(futures_train), desc="Loading training data"):
            feature_train, label_train = future_train.result()
            train_features.append(feature_train)
            train_labels.append(label_train)

    with ThreadPoolExecutor() as executor:
        futures_test = [executor.submit(process_file, file_name) for file_name in test_files]

        for i, future in tqdm(enumerate(futures_test), desc="Loading test data"):
            features, label = future.result()
            test_features.append(features)
            test_labels.append(label)

    # Convert the lists to NumPy arrays

    test_data = np.array(test_features)
    test_labels = np.array(test_labels).flatten()
    # Flatten test_data
    test_data = test_data.reshape(test_data.shape[0], -1)  # Reshape to (n_samples, 384)

    train_data = np.array(train_features)
    train_labels = np.array(train_labels).flatten()
    # Flatten test_data
    train_data = train_data.reshape(train_data.shape[0], -1)

    return train_data, train_labels, test_data, test_labels


def perform_knn(train_data, train_labels, test_data, test_labels, save_dir):
    # Define a range of values for n_neighbors to search
    n_neighbors_values = [1, 20]

    metrics_dict = {}
    os.makedirs(save_dir, exist_ok=True)
    accuracy_1 = 0
    balanced_acc_1 = 0
    weighted_f1_1 = 0

    accuracy_20 = 0
    balanced_acc_20 = 0
    weighted_f1_20 = 0
    for n_neighbors in n_neighbors_values:
        # Initialize a KNeighborsClassifier with the current n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        #flatten the test data, it a numpy array cast to float
        test_data_knn = test_data.reshape(test_data.shape[0], -1).astype(float)
        test_labels_knn = test_labels.reshape(-1)
        # Fit the KNN classifier to the training data
        knn.fit(train_data, train_labels)
        print("fitting done")
        print("X.shape", train_data.shape)
        print("y.shape", train_labels.shape)

        print("X_test.shape", test_data.shape)
        print("y_test.shape", test_labels.shape)

        print("Classes in test set: ", np.unique(test_labels))
        print("Classes in train set: ", np.unique(train_labels))
        # Predict labels for the test data
        test_predictions = knn.predict(test_data_knn)

        # Evaluate the classifier
        accuracy = accuracy_score(test_labels, test_predictions)
        balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
        weighted_f1 = f1_score(test_labels, test_predictions, average="weighted")

        if n_neighbors == 1:
            accuracy_1 = accuracy
            balanced_acc_1 = balanced_acc
            weighted_f1_1 = weighted_f1
        elif n_neighbors == 20:
            accuracy_20 = accuracy
            balanced_acc_20 = balanced_acc
            weighted_f1_20 = weighted_f1

        print(f"n_neighbors = {n_neighbors}")

        ## Calculate the classification report
        report = classification_report(test_labels, test_predictions, output_dict=True)

        print(f"report: {report}")

        current_metrics = {"accuracy": accuracy, "balanced_accuracy": balanced_acc, "weighted_f1": weighted_f1}
        print(current_metrics)
        # Store the metrics dictionary in the metrics_dict with a key indicating the number of neighbors
        metrics_dict[f"knn_{n_neighbors}"] = current_metrics
        # Convert the report to a Pandas DataFrame for logging
        # report_df = pd.DataFrame(report).transpose()

        # Log the final loss, accuracy, and classification report using wandb.log
        # wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
        filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
        file_path = os.path.join(save_dir, filename)
        # Speichern des DataFrames in der CSV-Datei
        df_labels_to_save.to_csv(file_path, index=False)

    return metrics_dict, accuracy_1, balanced_acc_1, weighted_f1_1, accuracy_20, balanced_acc_20, weighted_f1_20


def create_umap(data, labels, save_dir,labels_names, filename_addon="train"):
    # Create a UMAP model and fit it to your data
    # reducer = umap.UMAP(random_state=42)
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    # Specify the directory for saving the images

    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Loop through different figure sizes
    size = (24, 16)  # Add more sizes as needed

    # Create a scatter plot with the specified size
    plt.figure(figsize=size, dpi=300)
    #if the labels are key value pairs, we need to extract the keys
    
   


    plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=args.point_size, cmap="spectral")
    #use the labels names for the legend
    keys = list(labels_names.keys())
    label_names = [keys[i] for i in labels]
    #replace remove the first 19 characters
    label_names = [x[19:] for x in label_names]
    
    cbar=plt.colorbar()
    cbar.set_ticklabels(keys)

    #plt.colorbar(label="Class")

    
    plt.title("UMAP")

    # Specify the filename with the size information
    image_filename = f"umap_visualization_{Path(save_dir).name}_{size[0]}x{size[1]}_{filename_addon}.png"

    # Save the UMAP visualization as an image in the specified directory
    plt.savefig(os.path.join(umap_dir, image_filename))
    im = Image.open(os.path.join(umap_dir, image_filename))
    return im



def create_umap_v2(data, labels, save_dir, labels_names, filename_addon="train", point_size=10):
    # Create a UMAP model and fit it to your data
    reducer = umap.UMAP() 
    umap_data = reducer.fit_transform(data) 

    # Specify the directory for saving the images 
    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Loop through different figure sizes 
    size = (12, 8)  # Add more sizes as needed 

    # Create a scatter plot with the specified size
    plt.figure(figsize=size, dpi=300) 
    
    # Define the colormap
    cmap = plt.get_cmap('tab20')

    # Ensure that the colors for labels are consistent with the colormap
    scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=point_size, cmap=cmap)

    # Map the unique labels back to the correct label names
    unique_labels = np.unique(labels)
    print("unique_labels", unique_labels)
    print("labels_names", labels_names)

    # Generate the legend colors based on the scatter plot color mapping
    legend_handles = []
    for label in unique_labels:
        # Get the color for the label
        color = scatter.cmap(scatter.norm(label))  # Get the exact color for the label
        label_name = list(labels_names.keys())[list(labels_names.values()).index(label)]
        legend_handles.append(mpatches.Patch(color=color, label=label_name))

    # Display the legend with consistent colors
    plt.legend(handles=legend_handles, loc='upper right')
    plt.title("UMAP")

    # Specify the filename with the size information
    image_filename = f"umap_visualization_{Path(save_dir).name}_{size[0]}x{size[1]}_{filename_addon}.png"

    # Save the UMAP visualization as an image in the specified directory
    plt.savefig(os.path.join(umap_dir, image_filename))
    im = Image.open(os.path.join(umap_dir, image_filename))
    return im



def train_and_evaluate_logistic_regression(train_data, train_labels, test_data, test_labels, save_dir, max_iter=1000):
    # Initialize wandb

    M = train_data.shape[1]
    C = len(np.unique(train_labels))
    l2_reg_coef = 100 / (M * C)

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(C=1 / l2_reg_coef, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")

    logistic_reg.fit(train_data, train_labels)

    # Evaluate the model on the test data
    test_predictions = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(train_data)
    loss = log_loss(train_labels, predicted_probabilities)
    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    weighted_f1 = f1_score(test_labels, test_predictions, average="weighted")
    report = classification_report(test_labels, test_predictions, output_dict=True)

    df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
    filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    df_labels_to_save.to_csv(file_path, index=False)

    predicted_probabilities_df = pd.DataFrame(
        predicted_probabilities, columns=[f"Probability Class {i}" for i in range(predicted_probabilities.shape[1])]
    )
    predicted_probabilities_filename = f"{Path(save_dir).name}_predicted_probabilities_test.csv"
    predicted_probabilities_file_path = os.path.join(save_dir, predicted_probabilities_filename)
    predicted_probabilities_df.to_csv(predicted_probabilities_file_path, index=False)



    # some prints
    print(f"Final Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"balanced accuracy: {balanced_acc}")
    print(f"weighted_f1: {weighted_f1}")
    print(report)

    # Log the final loss, accuracy, and classification report using wandb.log
    final_loss = loss

    return loss, accuracy, balanced_acc, weighted_f1, report


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
