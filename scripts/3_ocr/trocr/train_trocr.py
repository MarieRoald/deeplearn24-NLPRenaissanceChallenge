import datasets

train_set = datasets.load_dataset(
    "imagefolder",
    data_dir="/hdd/home/mariero/deeplearn24/data/2_bounding_box/Doc-UFCN_processed",
    split="train",
)
