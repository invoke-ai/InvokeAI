from typing import Iterable

INDEX_PLACEHOLDER = "index_placeholder"


# Type alias for a 'ParsingTree', which is a recursive dict with string keys.
ParsingTree = dict[str, "ParsingTree"]


def insert_periods_into_kohya_key(key: str, parsing_tree: ParsingTree) -> str:
    """Insert periods into a Kohya key based on a parsing tree.

    Kohya format keys are produced by replacing periods with underscores in the original key.

    Example:
    ```
    key = "module_a_module_b_0_attn_to_k"
    parsing_tree = {
        "module_a": {
            "module_b": {
                INDEX_PLACEHOLDER: {
                    "attn": {},
                },
            },
        },
    }
    result = insert_periods_into_kohya_key(key, parsing_tree)
    > "module_a.module_b.0.attn.to_k"
    ```
    """
    result_parts = _split_kohya_key(key.split("_"), parsing_tree)
    if result_parts is None:
        raise ValueError(f"Key {key} does not match parsing tree {parsing_tree}.")
    return ".".join(result_parts)


def _split_kohya_key(parts: list[str], parsing_tree: ParsingTree) -> list[str] | None:
    """Splits underscore-separated key parts into the module names of a parsing tree, or returns None if they do not
    fit it.

    The shortest module name that fits wins, and a longer one is tried when the rest of the key does not fit below
    the shorter one: `ff_context_net_0_proj` is `ff_context`, though `ff` is a sibling in the tree too.
    """
    # A leading empty part (a double underscore in the key) belongs to no name.
    while parts and not parts[0]:
        parts = parts[1:]
    if not parts:
        return []
    for end in range(1, len(parts) + 1):
        name = "_".join(parts[:end])
        if name in parsing_tree:
            subtree = parsing_tree[name]
        elif end == 1 and name.isnumeric() and INDEX_PLACEHOLDER in parsing_tree:
            subtree = parsing_tree[INDEX_PLACEHOLDER]
        else:
            continue
        rest = _split_kohya_key(parts[end:], subtree)
        if rest is not None:
            return [name, *rest]
    return None


def kohya_module_path_is_leaf(module_path: str, parsing_tree: ParsingTree) -> bool:
    """True if a dotted module path walks the tree all the way to a leaf.

    ``insert_periods_into_kohya_key`` only rejects *leftover* tokens, so a prefix of a real path (e.g.
    ``blocks.0.attn``) parses cleanly without naming a module. Requiring a leaf rejects those.
    """
    subtree = parsing_tree
    for component in module_path.split("."):
        # Mirror ``insert_periods_into_kohya_key``'s precedence: an exact match wins over the index
        # placeholder. Without that, a numeric component would always be looked up as INDEX_PLACEHOLDER and
        # a tree enumerating the specific indices it accepts (e.g. Krea-2's ``tmlp``) could never reach its leaves.
        if component in subtree:
            subtree = subtree[component]
        elif component.isnumeric() and INDEX_PLACEHOLDER in subtree:
            subtree = subtree[INDEX_PLACEHOLDER]
        else:
            return False
    return not subtree


def generate_kohya_parsing_tree_from_keys(keys: Iterable[str]) -> ParsingTree:
    """Generate a parsing tree from a list of keys.

    Example:
    ```
    keys = [
        "module_a.module_b.0.attn.to_k",
        "module_a.module_b.1.attn.to_k",
        "module_a.module_c.proj",
    ]

    tree = generate_kohya_parsing_tree_from_keys(keys)
    > {
    >     "module_a": {
    >         "module_b": {
    >             INDEX_PLACEHOLDER: {
    >                 "attn": {
    >                     "to_k": {},
    >                     "to_q": {},
    >                 },
    >             }
    >         },
    >         "module_c": {
    >             "proj": {},
    >         }
    >     }
    > }
    ```
    """
    tree: ParsingTree = {}
    for key in keys:
        subtree: ParsingTree = tree
        for module_name in key.split("."):
            key = module_name
            if module_name.isnumeric():
                key = INDEX_PLACEHOLDER

            if key not in subtree:
                subtree[key] = {}

            subtree = subtree[key]
    return tree
