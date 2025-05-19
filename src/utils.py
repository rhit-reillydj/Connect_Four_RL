class dotdict(dict):
    """
    Dictionary that allows access to keys as attributes.
    Example: d.key instead of d['key']
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys()) 