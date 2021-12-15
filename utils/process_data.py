import sys, os, joblib
sys.path.append('.')

from config import datasets_path, db_path

import numpy as np

def process_datasets(path):

    poses = []
    translations = []

    datasets = [os.path.join(path, x) for x in os.listdir(path)]
    for dataset in datasets:
        print("Processing %s"%dataset)
        sequences = [os.path.join(dataset, x) for x in os.listdir(dataset)]
        for seq in sequences:
            files = [os.path.join(seq, x) for x in os.listdir(seq)]
            for file in files:
                x = np.load(file)
                pose = x['poses']
                trans = x['trans']
                poses.append(pose)
                translations.append(trans)

    poses = np.concatenate(poses)
    translations = np.concatenate(translations)

    print("shapes : ", poses.shape, translations.shape)

    joblib.dump({"pose":poses, "trans":translations}, os.path.join(db_path, "database.pt"))

if __name__ == '__main__':
    process_datasets(datasets_path)