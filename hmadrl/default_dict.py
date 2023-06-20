class DefaultDict(dict):

    def __getitem__(self, item):
        result = super().get(item, {})
        if type(result) == dict:
            return DefaultDict(result)
        return result
