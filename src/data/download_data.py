from datasets import load_dataset
import pandas as pd

dataset = load_dataset("mteb/banking77")

for split in dataset.keys():  # ['train', 'test']
    df = dataset[split].to_pandas()
    df.to_csv(f'data/raw/{split}.csv', index=False, encoding='utf-8')
    print(f"Done {split}.csv sauvegardé")