# Custom Nodes / Node Packs

Copy your node packs to this directory.

When nodes are added or changed, you must restart the app to see the changes.

## Directory Structure

For a node pack to be loaded, it must be placed in a directory alongside this
file. Here's an example structure:

```py
.
├── __init__.py # Invoke-managed custom node loader
│
├── cool_node
│   ├── __init__.py # see example below
│   └── cool_node.py
│
└── my_node_pack
    ├── __init__.py # see example below
    ├── tasty_node.py
    ├── bodacious_node.py
    ├── utils.py
    └── extra_nodes
        └── fancy_node.py
```

## Node Pack `__init__.py`

Each node pack must have an `__init__.py` file that imports its nodes.

The structure of each node or node pack is otherwise not important.

Here are examples, based on the example directory structure.

### `cool_node/__init__.py`

```py
from .cool_node import CoolInvocation
```

### `my_node_pack/__init__.py`

```py
from .tasty_node import TastyInvocation
from .bodacious_node import BodaciousInvocation
from .extra_nodes.fancy_node import FancyInvocation
```

Only nodes imported in the `__init__.py` file are loaded.
