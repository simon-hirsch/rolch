from .path import PathMethod


class ElasticNetPathMethod(PathMethod):
    """
    Exactly the same as PathMethod, but exposes alpha in the signature.
    """

    def __init__(self, alpha: float = 1.0, *args, **kwargs):
        super().__init__(alpha=alpha, *args, **kwargs)
