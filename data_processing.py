from datasets import Dataset

def prepare_preference_dataset(preference_pairs, images):
    """
    Prepare dataset for DPO training.
    """
    dataset_dict = {
        "image": [],
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    
    for pair, image in zip(preference_pairs, images):
        dataset_dict["image"].append(image)
        dataset_dict["prompt"].append(pair["question"])
        dataset_dict["chosen"].append(pair["chosen"])
        dataset_dict["rejected"].append(pair["rejected"])
    
    return Dataset.from_dict(dataset_dict)