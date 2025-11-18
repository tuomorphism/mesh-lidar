from urbaning.data import download_one_sequence, unzip_dataset

folder = "datasets/UrbanIng-V2X"
# download_one_sequence(
#    download_dir=folder
# )  # to download only one sequence for quick start purposes - optionally pass a sequence_name

unzip_dataset(folder)
