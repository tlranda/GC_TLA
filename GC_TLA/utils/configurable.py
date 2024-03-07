"""
    Class that reserves a slot to be configured via the _configure() classmethod,
    permitting updates to propagate to every instanced object (after the call, previously
    instanced objects are not updated)
"""
class Configurable():
    _configurable_core = {}
    @classmethod
    def get_mro_name(cls):
        """
            Necessary to ensure that different subclasses do not clobber one another's namespaces
        """
        mro = cls.mro()[0]
        mro_name = f"{mro.__module__}.{mro.__name__}"
        if mro_name not in cls._configurable_core.keys():
            cls._configurable_core[mro_name] = dict()
        return mro_name

    def __init__(self):
        # Only handle the core items, subclasses will define anything else
        for (k,v) in self._configurable_core[self.get_mro_name()].items():
            setattr(self,k,v)

    @classmethod
    def _configure(cls, **kwargs):
        cls._configurable_core[cls.get_mro_name()].update(kwargs)

    @classmethod
    def _remove_configure(cls, *args):
        for key in args:
            del cls._configurable_core[cls.get_mro_name()][key]

    def _update_from_core(self, **kwargs):
        key = self.get_mro_name()
        self.__class__._configurable_core[key].update(kwargs)
        for (k,v) in self._configurable_core[key].items():
            setattr(self, k, v)

    def _remove_from_core(self, *args):
        key = self.get_mro_name()
        for subkey in args:
            del self.__class__._configurable_core[key][subkey]
            delattr(self,subkey)

    def __str__(self):
        return "FactoryConfigurable("+\
               "; ".join([f"{k}:{getattr(self,k)}" for (k) in dir(self) if not k.startswith('_') and not callable(getattr(self,k))])+\
               ")"

