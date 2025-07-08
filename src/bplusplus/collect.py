import os
import random
import threading
from enum import Enum
from typing import Any, Dict, List, Optional

import pygbif
import requests
import validators
from tqdm import tqdm


#this lists currently supported groupings, more can be added with proper testing
class Group(str, Enum):
    scientificName="scientificName"

#TODO add back support for fetching from dataset (or csvs)
def collect(group_by_key: Group, search_parameters: dict[str, Any], images_per_group: int, output_directory: str, num_threads: int):

    groups: list[str] = search_parameters[group_by_key.value]

    # check if user wants to parallelize the process
    if num_threads > 1: 
        __threaded_collect(
            images_per_group=images_per_group, 
            output_directory=output_directory, 
            num_threads=num_threads,
            groups=groups)
    else: 
        __single_collect(
            search_parameters=search_parameters, 
            images_per_group=images_per_group, 
            output_directory=output_directory, 
            group_by_key=group_by_key,
            groups=groups,
        )

def __single_collect(group_by_key: Group, search_parameters: dict[str, Any], images_per_group: int, output_directory: str, groups: list[str]):

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
        for occurrence in tqdm(sampled_occurrences, desc=f"Downloading images for {group}", unit="image"):
            # image_url = occurrence.image_url.replace("original", "large") # hack to get max 1024px image

            __down_image(
                url=occurrence.image_url,
                group=group,
                ID_name=occurrence.key,
                folder=output_directory
            )
    
    print("Finished collecting images.")

# threaded_collect: paralellize the collection of images
def __threaded_collect(images_per_group: int, output_directory: str, num_threads: int, groups: list[str]):
    # Handle edge case where num_threads is greater than number of groups
    if num_threads >= len(groups):
        num_threads = len(groups)

    # Divide the species list into num_threads parts
    chunk_size = len(groups) // num_threads
    species_chunks = [
        groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)
    ]
    
    # Ensure we have exactly num_threads chunks (the last chunk might be larger if len(species_list) % num_threads != 0)
    while len(species_chunks) < num_threads:
        species_chunks.append([])
    
    threads = []
    for i, chunk in enumerate(species_chunks):
        thread = threading.Thread(
            target=__collect_subset,
            args=(chunk, images_per_group, output_directory, i)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("All collection threads have finished.")
    

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
        parameters["basisOfRecord"] = ["HUMAN_OBSERVATION", "LIVING_SPECIMEN", "MACHINE_OBSERVATION", "OBSERVATION", "OCCURRENCE"]
        parameters["lifeStage"] = ["Adult"]
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

def __collect_subset(species_subset: List[str], images_per_group: int, output_directory: str, thread_id: int):
    search_subset: Dict[str, Any] = {
        "scientificName": species_subset
    }
    
    print(f"Thread {thread_id} starting collection for {len(species_subset)} species.")
    
    __single_collect(
        search_parameters=search_subset,
        images_per_group=images_per_group,
        output_directory=output_directory,
        group_by_key=Group.scientificName,
        groups=species_subset
    )
    
    print(f"Thread {thread_id} finished collection.")




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