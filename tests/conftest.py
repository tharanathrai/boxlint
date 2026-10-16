import importlib.util
import os
import sys

PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make the plugin root importable so tests can `import checks`
sys.path.insert(0, PLUGIN_ROOT)


def load_plugin_package(name="lsc"):
    """Imports the plugin root as package ``name`` so modules with relative
    imports (``engine``, the operators) load as they do inside FiftyOne.
    """
    if name in sys.modules:
        return sys.modules[name]

    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(PLUGIN_ROOT, "__init__.py"),
        submodule_search_locations=[PLUGIN_ROOT],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
