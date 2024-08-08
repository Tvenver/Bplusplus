import csv
import os

import pandas as pd
import requests

#Step 0: create folders to store images inside, names are based on a .csv file where each row contains one specie name

def create_folders_from_csv(csv_file, directory_path):
    # Check if the folder path exists, if not, create it
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Open the CSV file and read the names
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip the header
        for row in csv_reader:
            name = row[0].strip()  # Assuming the name is in the first column
            print(name)
            folder_name = os.path.join(directory_path, name)
            # Create a folder with the name from the CSV
            os.makedirs(folder_name, exist_ok=True)


def collect_images(names_file: str, occurrence_file: str, multimedia_file:str, output_directory: str):
    # Provide the path to your CSV file and the folder where you want to create subfolders

    create_folders_from_csv(names_file, output_directory)

    # Step 1: Filtering the occurence dataset to only include species of interest, download images afterwards
    #set variables
    batch_size = 100000

    #specifiy path to occurence.txt file
    csv_reader = pd.read_table(occurrence_file, chunksize=batch_size)
    folders = [f for f in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, f))]
    col = ['gbifID', 'species']
    final_df = pd.DataFrame()  # Initialize the final DataFrame

    # Iterate through the batches
    for batch_df in csv_reader:
        batch_df = batch_df[col]
        batch_df = batch_df[batch_df['species'].isin(folders)]
        # Example: Print the first few rows of each batch to double check
        print(batch_df.head())
        final_df = pd.concat([final_df, batch_df], ignore_index=True)

    #output final df to csv, where csv contains link to multimedia and species
    final_df.to_csv(os.path.join(output_directory, 'occurence_filtered.csv'), index=False)

    #Step 1.b.: Load in links to multimedia, leftjoin with filtered occurence data
    #df1 = pd.read_table(os.path.join('data','multimedia.txt'), chunksize=batch_size)
    df1 = pd.read_table(multimedia_file, chunksize=batch_size)
    folders = [f for f in os.listdir(output_directory) if os.path.isdir(os.path.join(output_directory, f))]
    df2 = final_df

    # Step 3: Iterate over the chunks and left join them
    final_df = pd.DataFrame()  # Initialize the final DataFrame

    for chunk in df1:
        # Perform left join on the chunks
        joined_chunk = left_join_chunk(chunk, df2, 'gbifID')
        filtered_df = joined_chunk[joined_chunk['species'].isin(folders)]
        print(filtered_df)

        # Append the joined chunk to the final DataFrame
        final_df = pd.concat([final_df, filtered_df], ignore_index=True)

    # Step 4: Save the final DataFrame to a new file or use it as needed
    final_df.to_csv(os.path.join(output_directory, 'filtered_insect_species.csv'), index=False)

    # Step 1.c.: Download images in new folders named according to your convention
    df = final_df
    # df = pd.read_csv('directory/to/csv/from/observ.org/photos/sampled_super_small.csv')
    df['ID_name'] = df.index + 1

    # Assuming df is your DataFrame
    print('Start sampling per group')
    sampled = df.groupby('species').apply(sample_minimum).reset_index(drop=True)
    sampled.to_csv(os.path.join(output_directory, 'sampled_super_small.csv'), index=False)
    df = sampled

    df.apply(lambda row: down_image(row['identifier'], row['species'], row['ID_name']), axis=1)

# Step 1: Define a function to perform the left join on a specific chunk
def left_join_chunk(chunk1, chunk2, key_column):
    return pd.merge(chunk1, chunk2, on=key_column, how='left')

#uncomment sampling function, to reduce the test size to 150 image minimum or XXX% of original included testset (in our case from 60k images to 12k images)
#df = pd.read_csv('directory/to/csv/from/observ.org/photos/sampled_super_small.csv')
def sample_minimum(group):
    # Sample a minimum of 150 images or the total number of images if less than 150
    return group.sample(n=min(150, len(group)), random_state=42)  # Added random_state for reproducibility

#function to download insect images
def down_image(url, species, ID_name):
    directory = os.path.join('data/dataset', f"{species}")
    os.makedirs(directory, exist_ok=True)
    image_response = requests.get(url)
    image_name = f"{species}{ID_name}.jpg"  # You can modify the naming convention as per your requirements
    image_path = os.path.join(directory, image_name)
    with open(image_path, "wb") as f:
        f.write(image_response.content)
    print(f"{species}{ID_name} downloaded successfully.")


