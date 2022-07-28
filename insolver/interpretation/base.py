from abc import ABC, abstractmethod


class InterpretBase(ABC):
    @abstractmethod
    def show_explanation(self, instance):
        raise NotImplementedError

    @abstractmethod
    def plot(self, figsize, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError
