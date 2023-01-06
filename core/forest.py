"""
aq.py
Author: Arkadiusz Nowacki
"""

from core.aq import AQ
import numpy as np
from tqdm import tqdm
from typing import Union

class Forest:
    def __init__(self, n_tree: int, train_x: np.ndarray, train_y: np.ndarray, percent: float = 0.63):

        self.trees = self.__create_forest_AQ(n_tree=n_tree, train_x=train_x, train_y=train_y, percent=percent)

    def __create_forest_AQ(self, n_tree: int, train_x: np.ndarray, train_y: np.ndarray, percent: float) -> list[AQ]:

        trees = []
        print(f"Starting train AQ")
        for _ in tqdm(range(n_tree)):
            tree = AQ()
            idx = np.random.randint(train_x.shape[0], size=int(train_x.shape[0]*percent)) #randomly select a percentage of data from the dataset
            tree.fit(train_x=train_x[idx,:], train_y=train_y[idx,:])
            trees.append(tree)
        print(f"Train complete")
        return trees

    def predict(self, complex: list) -> Union[np.ndarray, float]:

        predict = []
        for tree in self.trees:
            predict.append(tree.predict(complex=complex).tolist())
        unique, counts = np.unique(np.array(predict), return_counts=True, axis=0) #choosing a prediction by voting
        counts = counts.tolist()
        return np.array(unique[counts.index(max(counts))]), max(counts)/len(self.trees)