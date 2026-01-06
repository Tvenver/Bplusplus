import os
import random
import signal
import sys
import threading
import time
import atexit
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pygbif
import requests
import validators
from tqdm import tqdm


# Currently supported groupings, more can be added with proper testing
class Group(str, Enum):
    scientificName = "scientificName"


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class CollectionProgress:
    """Thread-safe tracker for collection progress across species."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Set[str] = set()
        self._completed: Set[str] = set()
        self._failed: Dict[str, str] = {}  # species -> error message
        self._active = False
    
    def start(self, groups: List[str]):
        """Initialize tracking for a collection run."""
        with self._lock:
            self._pending = set(groups)
            self._completed = set()
            self._failed = {}
            self._active = True
    
    def mark_completed(self, group: str):
        """Mark a species as successfully completed."""
        with self._lock:
            self._pending.discard(group)
            self._completed.add(group)
            if group in self._failed:
                del self._failed[group]
    
    def mark_failed(self, group: str, error: str):
        """Mark a species as failed with error message."""
        with self._lock:
            self._failed[group] = error
    
    def get_incomplete(self) -> List[str]:
        """Get list of species not yet completed."""
        with self._lock:
            return list(self._pending - self._completed)
    
    def get_failed(self) -> Dict[str, str]:
        """Get dict of failed species and their errors."""
        with self._lock:
            return dict(self._failed)
    
    def is_active(self) -> bool:
        """Check if collection is active."""
        with self._lock:
            return self._active
    
    def finish(self):
        """Mark collection as finished."""
        with self._lock:
            self._active = False
    
    def print_status(self):
        """Print current collection status."""
        with self._lock:
            incomplete = list(self._pending - self._completed)
            if not incomplete and not self._failed:
                return
            
            print("\n" + "=" * 60)
            print("COLLECTION STATUS")
            print("=" * 60)
            print(f"Completed: {len(self._completed)}")
            print(f"Incomplete: {len(incomplete)}")
            print(f"Failed: {len(self._failed)}")
            
            if incomplete:
                print("\n⚠️  INCOMPLETE SPECIES (not yet processed):")
                for species in sorted(incomplete):
                    print(f"  - {species}")
            
            if self._failed:
                print("\n❌ FAILED SPECIES (errors encountered):")
                for species, error in sorted(self._failed.items()):
                    print(f"  - {species}: {error}")
            
            print("=" * 60)


# Global progress tracker
_progress = CollectionProgress()


def _print_status_on_exit():
    """Print incomplete species on exit."""
    if _progress.is_active():
        _progress.print_status()


def _signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print("\n\n⚠️  Collection interrupted by user!")
    _progress.print_status()
    sys.exit(1)


# Register exit handlers
atexit.register(_print_status_on_exit)
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# RETRY CONFIGURATION
# ============================================================================

DEFAULT_RETRY_CONFIG = {
    "max_retries": 5,
    "initial_wait": 10,      # seconds
    "max_wait": 300,         # 5 minutes max
    "backoff_factor": 2,     # exponential backoff
}


# Default quality filters for high-quality training data from GBIF
# Reference: https://www.gbif.org/developer/occurrence
DEFAULT_QUALITY_FILTERS = {
    # Media filters
    "mediaType": ["StillImage"],
    
    # Basis of record - observation types most likely to have quality images
    "basisOfRecord": [
        "HUMAN_OBSERVATION",
        "MACHINE_OBSERVATION",
        "OBSERVATION",
    ],
    
    # Life stage - adult insects for consistent morphology
    "lifeStage": ["Adult"],
    
    # Occurrence status - only confirmed presence records
    "occurrenceStatus": "PRESENT",
    
    # # Geospatial quality - ensure valid coordinates without issues
    # "hasCoordinate": True,
    # "hasGeospatialIssue": False,
    
    # # Coordinate precision - max 1km uncertainty for reliable location
    # "coordinateUncertaintyInMeters": "0,1000",
    
    # # License - permissive licenses for research/training use
    # # CC0_1_0: Public domain, CC_BY_4_0: Attribution only, CC_BY_NC_4_0: Non-commercial
    # "license": ["CC0_1_0", "CC_BY_4_0", "CC_BY_NC_4_0"],
    
    # Year range - recent records tend to have better quality images
    # Can be overridden by user
    "year": "2010,2025",
}

def collect(
    group_by_key: Group,
    search_parameters: dict[str, Any],
    images_per_group: int,
    output_directory: str,
    num_threads: int,
    use_quality_filters: bool = True,
    quality_filter_overrides: Optional[dict[str, Any]] = None,
    max_retries: int = 5,
    initial_wait: int = 10,
):
    """
    Collect images from GBIF for training data.
    
    Args:
        group_by_key: How to group occurrences (e.g., by scientificName)
        search_parameters: GBIF search parameters including species list
        images_per_group: Number of images to download per group
        output_directory: Directory to save downloaded images
        num_threads: Number of parallel download threads
        use_quality_filters: Apply default quality filters for training data
        quality_filter_overrides: Override specific quality filter values
        max_retries: Maximum retry attempts for failed API calls
        initial_wait: Initial wait time in seconds before retry (doubles each retry)
    
    On interruption or failure, prints list of incomplete species.
    """
    groups: list[str] = search_parameters[group_by_key.value]
    
    # Initialize progress tracking
    _progress.start(groups)
    
    # Build retry config
    retry_config = {
        "max_retries": max_retries,
        "initial_wait": initial_wait,
        "max_wait": 300,
        "backoff_factor": 2,
    }
    
    # Build quality filters
    quality_filters = {}
    if use_quality_filters:
        quality_filters = DEFAULT_QUALITY_FILTERS.copy()
        if quality_filter_overrides:
            quality_filters.update(quality_filter_overrides)
        print("Quality filters enabled:")
        for key, value in quality_filters.items():
            print(f"  {key}: {value}")
    
    print(f"\nStarting collection for {len(groups)} species...")
    print(f"Retry config: max_retries={max_retries}, initial_wait={initial_wait}s\n")
    
    try:
        # Check if user wants to parallelize the process
        if num_threads > 1:
            __threaded_collect(
                images_per_group=images_per_group,
                output_directory=output_directory,
                num_threads=num_threads,
                groups=groups,
                quality_filters=quality_filters,
                retry_config=retry_config,
            )
        else:
            __single_collect(
                search_parameters=search_parameters,
                images_per_group=images_per_group,
                output_directory=output_directory,
                group_by_key=group_by_key,
                groups=groups,
                quality_filters=quality_filters,
                retry_config=retry_config,
            )
    finally:
        _progress.finish()
        _progress.print_status()

def __single_collect(
    group_by_key: Group,
    search_parameters: dict[str, Any],
    images_per_group: int,
    output_directory: str,
    groups: list[str],
    quality_filters: dict[str, Any],
    retry_config: dict[str, Any],
):
    """Single-threaded collection of images with retry logic."""
    __create_folders(names=groups, directory=output_directory)

    print("Beginning to collect images from GBIF...")
    for group in groups:
        success = __collect_single_group(
            group=group,
            group_by_key=group_by_key,
            search_parameters=search_parameters.copy(),
            images_per_group=images_per_group,
            output_directory=output_directory,
            quality_filters=quality_filters,
            retry_config=retry_config,
        )
        if success:
            _progress.mark_completed(group)

    print("Finished collecting images.")


def __collect_single_group(
    group: str,
    group_by_key: Group,
    search_parameters: dict[str, Any],
    images_per_group: int,
    output_directory: str,
    quality_filters: dict[str, Any],
    retry_config: dict[str, Any],
) -> bool:
    """
    Collect images for a single group with retry logic.
    
    Returns:
        bool: True if successful, False if all retries exhausted
    """
    max_retries = retry_config["max_retries"]
    initial_wait = retry_config["initial_wait"]
    max_wait = retry_config["max_wait"]
    backoff_factor = retry_config["backoff_factor"]
    
    for attempt in range(max_retries + 1):
        try:
            # Fetch occurrences
            occurrences_json = _fetch_occurrences(
                group_key=group_by_key,
                group_value=group,
                parameters=search_parameters.copy(),
                quality_filters=quality_filters,
                totalLimit=10000,
            )
            optional_occurrences = map(lambda x: __parse_occurrence(x), occurrences_json)
            occurrences = list(filter(None, optional_occurrences))

            if not occurrences:
                print(f"⚠️  No valid occurrences found for {group}")
                return True  # Not a failure, just no data

            random.seed(42)  # for reproducibility
            sampled_occurrences = random.sample(occurrences, min(images_per_group, len(occurrences)))

            print(f"Downloading {len(sampled_occurrences)} images into the {group} folder...")
            
            # Download images with individual retry
            for occurrence in tqdm(sampled_occurrences, desc=f"{group}", unit="img"):
                __download_with_retry(
                    url=occurrence.image_url,
                    group=group,
                    ID_name=occurrence.key,
                    folder=output_directory,
                    max_retries=3,
                )
            
            print(f"✓ Completed: {group}")
            return True
            
        except (requests.exceptions.Timeout, 
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                Exception) as e:
            
            error_msg = str(e)[:100]
            _progress.mark_failed(group, error_msg)
            
            if attempt < max_retries:
                wait_time = min(initial_wait * (backoff_factor ** attempt), max_wait)
                print(f"\n⚠️  Error for {group}: {error_msg}")
                print(f"   Retry {attempt + 1}/{max_retries} in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Failed after {max_retries} retries: {group}")
                print(f"   Error: {error_msg}")
                return False
    
    return False


def __download_with_retry(url: str, group: str, ID_name: str, folder: str, max_retries: int = 3):
    """Download a single image with retry logic."""
    for attempt in range(max_retries + 1):
        try:
            __down_image(url=url, group=group, ID_name=ID_name, folder=folder)
            return
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # Quick exponential backoff for images
            else:
                # Silent fail for individual images - don't halt the whole process
                pass

def __threaded_collect(
    images_per_group: int,
    output_directory: str,
    num_threads: int,
    groups: list[str],
    quality_filters: dict[str, Any],
    retry_config: dict[str, Any],
):
    """Parallelize the collection of images across multiple threads."""
    # Handle edge case where num_threads is greater than number of groups
    if num_threads >= len(groups):
        num_threads = len(groups)

    # Divide the species list into num_threads parts
    chunk_size = len(groups) // num_threads
    species_chunks = [
        groups[i : i + chunk_size] for i in range(0, len(groups), chunk_size)
    ]

    # Ensure we have exactly num_threads chunks
    while len(species_chunks) < num_threads:
        species_chunks.append([])

    threads = []
    for i, chunk in enumerate(species_chunks):
        thread = threading.Thread(
            target=__collect_subset,
            args=(chunk, images_per_group, output_directory, i, quality_filters, retry_config),
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("All collection threads have finished.")
    

def _fetch_occurrences(
    group_key: str,
    group_value: str,
    parameters: dict[str, Any],
    quality_filters: dict[str, Any],
    totalLimit: int,
) -> list[dict[str, Any]]:
    """Fetch occurrences from GBIF with quality filters applied."""
    parameters[group_key] = group_value
    return __next_batch(
        parameters=parameters,
        quality_filters=quality_filters,
        total_limit=totalLimit,
        offset=0,
        current=[],
    ) 

def __next_batch(
    parameters: dict[str, Any],
    quality_filters: dict[str, Any],
    total_limit: int,
    offset: int,
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recursively fetch batches of occurrences from GBIF."""
    # Build search parameters
    search_params = {**parameters}
    search_params["limit"] = total_limit
    search_params["offset"] = offset
    
    # Apply quality filters
    search_params.update(quality_filters)
    
    search = pygbif.occurrences.search(**search_params)
    occurrences = search["results"]
    
    if search["endOfRecords"] or len(current) >= total_limit:
        return current + occurrences
    else:
        new_offset = search["offset"]
        count = search["limit"]
        return __next_batch(
            parameters=parameters,
            quality_filters=quality_filters,
            total_limit=total_limit,
            offset=new_offset + count,
            current=current + occurrences,
        )

# Function to download insect images
def __down_image(url: str, group: str, ID_name: str, folder: str, timeout: int = 30):
    """Download a single image with timeout."""
    directory = os.path.join(folder, f"{group}")
    os.makedirs(directory, exist_ok=True)
    image_response = requests.get(url, timeout=timeout)
    image_response.raise_for_status()  # Raise on bad status codes
    image_name = f"{group}{ID_name}.jpg"
    image_path = os.path.join(directory, image_name)
    with open(image_path, "wb") as f:
        f.write(image_response.content)

def __create_folders(names: list[str], directory: str):
    print("Creating folders for images...")
    # Check if the folder path exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name in names:
        folder_name = os.path.join(directory, name)
        # Create a folder using the group name
        os.makedirs(folder_name, exist_ok=True)

def __collect_subset(
    species_subset: List[str],
    images_per_group: int,
    output_directory: str,
    thread_id: int,
    quality_filters: Dict[str, Any],
    retry_config: Dict[str, Any],
):
    """Worker function for threaded collection."""
    search_subset: Dict[str, Any] = {"scientificName": species_subset}

    print(f"Thread {thread_id} starting collection for {len(species_subset)} species.")

    __single_collect(
        search_parameters=search_subset,
        images_per_group=images_per_group,
        output_directory=output_directory,
        group_by_key=Group.scientificName,
        groups=species_subset,
        quality_filters=quality_filters,
        retry_config=retry_config,
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