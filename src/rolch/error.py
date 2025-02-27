class OutOfSupportError(ValueError):
    """Exception raised for values that are out of support."""

    def __init__(self, message="This value is out of support."):
        self.message = message
        super().__init__(self.message)
