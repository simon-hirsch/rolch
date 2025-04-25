from .elasticnet import ElasticNetPathMethod


class LassoPathMethod(ElasticNetPathMethod):
    def __init__(self, *args, **kwargs):
        super().__init__(alpha=1.0, *args, **kwargs)
