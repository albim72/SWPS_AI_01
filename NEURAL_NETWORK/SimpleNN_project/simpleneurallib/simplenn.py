import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        np.random.seed(42)
        self.weights = 2*np.random.random((3,1))-1

    def __repr__(self):
        return (f"nowa sieć neuronowa oparta na klasie: {self.__class__.__name__}\n"
                f"Wylosowano wagi:\n{self.weights}\n")
