import csv
import os
import random

import pandas as pd
import pygbif
import requests


class Occurrence:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

#TODO add back support for fetching from dataset (or csvs)
def collect_images(scientificNames: list[str], images_per_species: int, output_directory: str):
    __create_folders(
        names=scientificNames,
        directory=output_directory
    )

    print("Beginning to collect images from GBIF...")
    for name in scientificNames:
        print(f"Collecting images for {name}")
        occurrences = _fetch_occurrences(scientificName=name, totalLimit=10000)
        print(f"{name} : {len(occurrences)} occurrences fetched, will sample for {images_per_species}")

        #TODO: filter for images then sample

        random.seed(42) # for reproducibility
        sampled_occurrences = random.sample(occurrences, min(images_per_species, len(occurrences)))
        
        for occurrence in sampled_occurrences:
            image_url: str = occurrence.media[0].get("identifier")
            if image_url is not None:
                image_url = image_url.replace("original", "large") # hack to get max 1024px image

                __down_image(
                    url=image_url,
                    species=name,
                    ID_name=occurrence.key
                )
            else:
                print(f"Image URL not found for {occurrence.key}...")


def _fetch_occurrences(scientificName: str, totalLimit: int) -> list[Occurrence]:
    return __next_batch_for_species(
        scientificName=scientificName,
        totalLimit=totalLimit,
        offset=0,
        current=[]
    ) 

def __next_batch_for_species(scientificName: str, totalLimit: int, offset: int, current: list[Occurrence]) -> list[Occurrence]:
        search = pygbif.occurrences.search(scientificName=scientificName, limit=totalLimit, offset=offset, mediaType=["StillImage"])
        results = search["results"]
        occurrences = list(map(lambda x: Occurrence(**x), results))
        if search["endOfRecords"] or len(current) >= totalLimit:
            return current + occurrences
        else:
            offset = search["offset"]
            count = search["limit"] # this seems to be returning the count, and `count` appears to be returning the total number of results returned by the search
            return __next_batch_for_species(
                scientificName=scientificName,
                totalLimit=totalLimit,
                offset=offset + count,
                current=current + occurrences
            )

#function to download insect images
def __down_image(url: str, species: str, ID_name: str):
    directory = os.path.join('data/dataset', f"{species}")
    os.makedirs(directory, exist_ok=True)
    image_response = requests.get(url)
    image_name = f"{species}{ID_name}.jpg"  # You can modify the naming convention as per your requirements
    image_path = os.path.join(directory, image_name)
    with open(image_path, "wb") as f:
        f.write(image_response.content)
    print(f"{image_name} downloaded successfully.")

def __create_folders(names: list[str], directory: str):
    print("Creating folders for images...")
    # Check if the folder path exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name in names:
        folder_name = os.path.join(directory, name)
        # Create a folder using the scientific name
        os.makedirs(folder_name, exist_ok=True)

