"""
Author: Nicolas Raymond

This file is used to store hyperparameter class
"""


class HP:
    """
    Defines an hyperparameter
    """
    def __init__(self, name: str, distribution: str):
        """
        Sets the name of the hp and the distribution the suggestion must be sampled from
        Args:
            name: name of the hyperparameter
            distribution: optuna distribution from which it must be sampled
        """
        self.name = name
        self.distribution = distribution

    def __repr__(self):
        return self.name


class CategoricalHP(HP):
    """
    Categorical hyperparameter
    """
    def __init__(self, name: str):
        """
        Sets attribute using parent's constructor

        Args:
            name: name of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.CATEGORICAL)


class NumericalIntHP(HP):
    """
    Numerical integer hyperparameter
    """
    def __init__(self, name: str):
        """
        Sets attribute using parent's constructor

        Args:
            name: name of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.INT)


class NumericalContinuousHP(HP):
    """
    Numerical continuous hyperparameter
    """
    def __init__(self, name: str):
        """
        Sets attribute using parent's constructor

        Args:
            name: name of the hyperparameter
        """
        super().__init__(name=name, distribution=Distribution.UNIFORM)


class Distribution:
    """
    Stores possible types of distribution
    """
    INT: str = "int"                # Int uniform
    UNIFORM: str = "uniform"
    CATEGORICAL: str = "categorical"
