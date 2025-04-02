import bplusplus
from typing import Any
from pathlib import Path

MAIN_DIR = Path("/mnt/nvme1n1p1/datasets/sample")

GBIF_DATA_DIR = MAIN_DIR / "GBIF_data"
PREPARED_DATA_DIR = MAIN_DIR / "prepared_data"
TRAINED_MODEL_DIR = MAIN_DIR / "trained_model"


# names = ["Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris"]

# search: dict[str, Any] = {
#     "scientificName": names
# }

# bplusplus.collect(
#     group_by_key=bplusplus.Group.scientificName,
#     search_parameters=search, 
#     images_per_group=50,
#     output_directory=GBIF_DATA_DIR,
#     num_threads=3
# )

bplusplus.prepare(
    input_directory=GBIF_DATA_DIR,
    output_directory=PREPARED_DATA_DIR,
    one_stage=False,
    with_background=False, # Set to False if you don't want to include/download background images
    size_filter=True, #set to True if you want to filter by size of insect 
    sizes=["large"] #set to list of sizes if you want to filter by size of insect 
)

