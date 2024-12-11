import os
import random
from enum import Enum
from typing import Any, Optional

import pygbif
import requests
import validators


#this lists currently supported groupings, more can be added with proper testing
class Group(str, Enum):
    scientificName="scientificName"

#TODO add back support for fetching from dataset (or csvs)
def collect(group_by_key: Group, search_parameters: dict[str, Any], images_per_group: int, output_directory: str):

    groups: list[str] = search_parameters[group_by_key.value]

    #TODO throw error if groups is not a str list

    __create_folders(
        names=groups,
        directory=output_directory
    )

    print("Beginning to collect images from GBIF...")
    for group in groups:
        # print(f"Collecting images for {group}...")
        occurrences_json = _fetch_occurrences(group_key=group_by_key, group_value=group, parameters=search_parameters, totalLimit=10000)
        optional_occurrences = map(lambda x: __parse_occurrence(x), occurrences_json)
        occurrences = list(filter(None, optional_occurrences))

        # print(f"{group} : {len(occurrences)} parseable occurrences fetched, will sample for {images_per_group}")

        random.seed(42) # for reproducibility
        sampled_occurrences = random.sample(occurrences, min(images_per_group, len(occurrences)))
        
        print(f"Downloading {len(sampled_occurrences)} images into the {group} folder...")
        for occurrence in sampled_occurrences:
            # image_url = occurrence.image_url.replace("original", "large") # hack to get max 1024px image

            __down_image(
                url=occurrence.image_url,
                group=group,
                ID_name=occurrence.key,
                folder=output_directory
            )
    
    print("Finished collecting images.")

def _fetch_occurrences(group_key: str, group_value: str, parameters: dict[str, Any], totalLimit: int) -> list[dict[str, Any]]:
    parameters[group_key] = group_value
    return __next_batch(
        parameters=parameters,
        total_limit=totalLimit,
        offset=0,
        current=[]
    ) 

def __next_batch(parameters: dict[str, Any], total_limit: int, offset: int, current: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parameters["limit"] = total_limit
        parameters["offset"] = offset
        parameters["mediaType"] = ["StillImage"]
        search = pygbif.occurrences.search(**parameters)
        occurrences = search["results"]
        if search["endOfRecords"] or len(current) >= total_limit:
            return current + occurrences
        else:
            offset = search["offset"]
            count = search["limit"] # this seems to be returning the count, and `count` appears to be returning the total number of results returned by the search
            return __next_batch(
                parameters=parameters,
                total_limit=total_limit,
                offset=offset + count,
                current=current + occurrences
            )

#function to download insect images
def __down_image(url: str, group: str, ID_name: str, folder: str):
    directory = os.path.join(folder, f"{group}")
    os.makedirs(directory, exist_ok=True)
    image_response = requests.get(url)
    image_name = f"{group}{ID_name}.jpg"  # You can modify the naming convention as per your requirements
    image_path = os.path.join(directory, image_name)
    with open(image_path, "wb") as f:
        f.write(image_response.content)
    # print(f"{image_name} downloaded successfully.")

def __create_folders(names: list[str], directory: str):
    print("Creating folders for images...")
    # Check if the folder path exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name in names:
        folder_name = os.path.join(directory, name)
        # Create a folder using the group name
        os.makedirs(folder_name, exist_ok=True)




class Occurrence:

    def __init__(self, key: str, image_url: str) -> None:
         self.key = key
         self.image_url = image_url
         

def __parse_occurrence(json: dict[str, Any]) -> Optional[Occurrence]:
    if (key := json.get("key", str)) is not None \
        and (image_url := json.get("media", {})[0].get("identifier", str)) is not None \
            and validators.url(image_url):
         
         return Occurrence(key=key, image_url=image_url)
    else:
         return None