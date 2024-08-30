import importlib
import pathlib
# Crawl subdirectories (that don't start with '.' or '_') as importable submodules
implemented = sorted([_.name for _ in pathlib.Path(__file__).parents[0].iterdir() if _.is_dir() and not (_.name.startswith('.') or _.name.startswith('_'))],
                     reverse=True)
del pathlib
__all__ = implemented

def __getattr__(name):
    for impl in implemented:
        # This mechanism may be somewhat brittle for matching, but currently it is sufficient
        # ie: Variations such as 'myimpl_abc' and 'myimpl_abcdef' would prefer abc and never import
        # abcdef (matching substrings problem + order of the implemented list)
        if not name.startswith(impl):
            continue
        if name == impl:
            module = importlib.import_module('.'+impl, 'GC_TLA.implementations')
            return module
        else:
            try:
                module = importlib.import_module('.'+impl, 'GC_TLA.implementations')
                return getattr(module, name)
            except Exception as e:
                # When debugging, rather than Attribute Error you probably want to re-raise the exception
                raise e
                #raise AttributeError
    # Default: Not handled, continue parsing
    raise AttributeError

