"""
    Class that reserves a slot to be configured via the _configure() classmethod,
    permitting updates to propagate to every instanced object (after the call, previously
    instanced objects are not updated)
"""
class FactoryConfigurable():
    _configurable_core = {}
    def __init__(self):
        # Only handle the core items, subclasses will define anything else
        for (k,v) in self._configurable_core.items():
            setattr(self,k,v)

    @classmethod
    def _configure(cls, **kwargs):
        cls._configurable_core.update(kwargs)

    def __str__(self):
        return "FactoryConfigurable("+\
               "; ".join([f"{k}:{getattr(self,k)}" for (k) in dir(self) if not k.startswith('_') and not callable(getattr(self,k))])+\
               ")"

