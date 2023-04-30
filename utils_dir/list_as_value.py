from typing import List, Any


class ListAsValue:

    def __init__(self, values: List[Any]):
        self.values = values

    def __getattr__(self, item):
        attributes = [getattr(value, item) for value in self.values]
        return ListAsValue(attributes)

    def __call__(self, *args, **kwargs):
        results = [value(*args, **kwargs) for value in self.values]
        return ListAsValue(results)