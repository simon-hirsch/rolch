from .path import PathMethod


class RidgePathMethod(PathMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(alpha=0.0, *args, **kwargs)
