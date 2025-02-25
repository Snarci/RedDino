import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="split generation")

parser.add_argument(
    "--dataset_path",
    help="path to dataset",
    default="",
    type=str,
)

parser.add_argument(
    "--dataset_name",
    help="dataset_name",
    default="",
    type=str,
)

parser.add_argument(
    "--output_path",
    help="output path",
    default="splits",
    type=str,
)


def create_slide_splits(folder_path, dataset_name):
    """
    Splits the data in the given folder into training and validation sets,
    ensuring class balance between the sets. Assumes the folder structure
    where the parent folder name is the class label.

    Parameters:
    - folder_path: str, the path to the folder containing the class folders.

    Creates two files:
    - 'train.csv': Contains paths and labels for the training set.
    - 'val.csv': Contains paths and labels for the validation set.
    """
    data = []

    # Iterate over each class directory in the folder
    for class_name in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, class_name)
        if os.path.isdir(class_dir):
            # For each image in the class directory
            for img in os.listdir(class_dir):
                if img.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
                    # Collect image path and class name
                    img_path = os.path.join(class_dir, img)
                    data.append((img_path, class_name))

    # Convert collected data into a DataFrame
    df = pd.DataFrame(data, columns=["Image Path", "Label"])
    #out folder creation
    out_folder = os.path.join(args.output_path, dataset_name)
    os.makedirs(out_folder, exist_ok=True)
    # Split the data slide-wise&patient-wise
    # after the "Slide" keyword the slide number is mentioned in the image path and split for each slide
    #example: Slide 1 (1,93,472,116,497).png
    numbers = df["Image Path"].str.extract(r"Slide (\d+)")[0].astype(int)
    df["Slide"] = numbers
    df = df.sort_values(by=["Slide"])
    print("Number of slides: ", len(df["Slide"].unique()))
    for slide in df["Slide"].unique():
        print(f"Slide {slide} has {len(df[df['Slide'] == slide])} images")
        slide_df = df[df["Slide"] == slide]
        slide_df = slide_df.drop(columns=["Slide"])
        slide_df = slide_df.reset_index(drop=True)

        out_file = os.path.join(out_folder, f"slide_{slide}.csv")
        slide_df.to_csv(out_file, index=False)

        


if __name__ == "__main__":
    args = parser.parse_args()
    create_slide_splits(args.dataset_path, args.dataset_name)
