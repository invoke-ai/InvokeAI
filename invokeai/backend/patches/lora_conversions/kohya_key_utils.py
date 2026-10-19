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
    # Split key into parts by underscore.
    parts = key.split("_")

    # Build up result by walking through parsing tree and parts.
    result_parts: list[str] = []
    current_part = ""
    current_tree = parsing_tree

    for part in parts:
        if len(current_part) > 0:
            current_part = current_part + "_"
        current_part += part

        if current_part in current_tree:
            # Match found.
            current_tree = current_tree[current_part]
            result_parts.append(current_part)
            current_part = ""
        elif current_part.isnumeric() and INDEX_PLACEHOLDER in current_tree:
            # Match found with index placeholder.
            current_tree = current_tree[INDEX_PLACEHOLDER]
            result_parts.append(current_part)
            current_part = ""

    if len(current_part) > 0:
        raise ValueError(f"Key {key} does not match parsing tree {parsing_tree}.")

    return ".".join(result_parts)


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
