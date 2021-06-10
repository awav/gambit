from os import listdir as _list_dir
import importlib as _importlib

_examples = "/".join(__loader__.path.split("/")[:-1])
_modules = [example.replace(".py", "") for example in _list_dir(_examples) if example[-3:] == ".py" and example != "__init__.py"]

def _load(name):
  spec = _importlib.util.spec_from_file_location(name, f"{_examples}/{name}.py")
  module = _importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

cases = [_load(name) for name in _modules]
