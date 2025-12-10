## Prepare
Install the package in local you can import the package like the package you install in your envirnment.
This need you construct a package dictionary and construct a package constructor.

## Small Project (A Simplest Package)
Suppose we have a `Python` package named `cosfunc.py`.
We first create a directory named (e.g. `cosfunc`, also you can set a other name, this will not affect the package name) to contain the package.
Then we create a file named `__init__.py` in the directory `cosfunc`.
In the file `__init__.py`, we add the following code:
```python
from cosfunc import *
```
The code `from cosfunc import *` means we import all the functions in the file `cosfunc.py` to the package `cosfunc`.

We then move the file `cosfunc.py` to the dictionary `CosFuncTion`.

The functions in the file `cosfunc.py` is all your package.

Usage:
```python
import cosfunc
```
You can use the functions in the package `cosfunc` like the following:
```python
cosfunc.function_name()
```

### Install the package
Our package is like this:
```
cosfunc/
    __init__.py
    cosfunc.py
```
we need a file named `pyproject.toml` in the directory `cosfunc`.
In the file `pyproject.toml`, we add the following code:
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cosfunc"
version = "0.0.7"
description = "A simple cosmological function package"
```
the name item in the `[project]` section is the name of the package.
It should be the same as the python package name (e.g. `cosfunc` to the python package `cosfunc.py`).
The version item in the `[project]` section is the version of the package. You can change or not change it.

After all process have done, you can install the package in local by running the command:
```bash
pip install -e .
```
in the directory `CosFuncTion`. (The -e means install the package in editable mode, so you can modify the package and the changes will be reflected in the package.)

## Large Project (A Package with Multiple modules)
If you have a large project, you can construct a subdirectory to contain the modules of the package.

For example, if you have a package named `cosfunc`, you can construct a subdirectory named `cosfunc/modules` to contain the modules of the package.
```
EoRCaLC/
    eorcalc/
        __init__.py
        powerspec.py
        ioninti_gpu.py
        reion_field.py
        special.py
    pyproject.toml
```
We can keep the file `__init__.py` in the directory `EoRCaLC/eorcalc` empty.
So we can import the modules in the package `EoRCaLC.eorcalc` like the following:
```python
import eorcalc.reion_field as rf
#use the function in the module reion_field
rf.function_name()
```
in the pyproject.toml, we add the following code:
```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "eorcalc"
version = "0.0.7"
description = "A simple cosmological function package"
```
``Notion:`` keep the `name` item in the `[project]` section the same as the directory name `EoRCaLC/eorcalc` (e.g. name = "eorcalc").

And then we can install the package in local by running the command:
```bash
pip install -e .
```
in the directory `EoRCaLC`.

<!-- 
For example, if you have a package named `cosfunc`, you can construct a subdirectory named `cosfunc/modules` to contain the modules of the package.
In the file `__init__.py` in the directory `cosfunc/modules`, you can add the following code:
```python
from .module1 import *
from .module2 import *
```
The code `from .module1 import *` means we import all the functions in the file `module1.py` to the package `cosfunc.modules.module1`.
The code `from .module2 import *` means we import all the functions in the file `module2.py` to the package `cosfunc.modules.module2`. -->



