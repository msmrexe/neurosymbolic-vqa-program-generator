"""
Utilities for converting, parsing, and handling program representations.
Programs can be represented as lists, trees, or strings in
prefix or postfix notation.
"""

from collections import deque

# --- Constants for Function Input Arity ---

# A set of all function names that take two inputs
_TWO_INPUT_FUNCS = {
    "equal_color",
    "equal_integer",
    "equal_material",
    "equal_shape",
    "equal_size",
    "union",
    "intersect",
    "less_than",
    "greater_than",
}


def get_num_inputs(func_node):
    """
    Determines the number of inputs a given function node requires.

    Args:
        func_node (dict or str): The function node (as a dict)
                                or its string representation.

    Returns:
        int: The number of inputs (0, 1, or 2).
    """
    if isinstance(func_node, str):
        func_node = str_to_function(func_node)

    name = func_node["function"]

    if name == "scene":
        return 0
    if name in _TWO_INPUT_FUNCS or "equal" in name:
        return 2
    return 1


# --- String-to-Function and Function-to-String ---


def function_to_str(func_node):
    """
    Converts a function node dict to its compact string representation.
    Example: {'function': 'filter_color', 'value_inputs': ['blue']}
          -> 'filter_color[blue]'
    """
    value_str = ""
    if func_node["value_inputs"]:
        values = ",".join(func_node["value_inputs"])
        value_str = f"[{values}]"
    return f"{func_node['function']}{value_str}"


def str_to_function(func_str):
    """
    Parses a function string into its dictionary representation.
    Example: 'filter_color[blue]'
          -> {'function': 'filter_color', 'value_inputs': ['blue']}
    """
    if "[" not in func_str:
        return {"function": func_str, "value_inputs": []}

    # Split the string at the first '[' and remove the trailing ']'
    name, value_part = func_str.split("[", 1)
    values = value_part[:-1].split(",")
    return {"function": name, "value_inputs": values}


def list_to_str(program_list):
    """
    Converts a list of function nodes into a single space-delimited string.
    """
    return " ".join(function_to_str(func) for func in program_list)


# --- List <-> Tree Conversions ---


def list_to_tree(program_list):
    """
    Converts a program from list representation (dependency-ordered)
    to a nested tree representation.
    """

    def _build_node(node_data):
        """Recursive helper to build a tree node."""
        return {
            "function": node_data["function"],
            "value_inputs": list(node_data["value_inputs"]),
            "inputs": [_build_node(program_list[i]) for i in node_data["inputs"]],
        }

    # The root of the tree is always the last element in the list
    if not program_list:
        return None
    return _build_node(program_list[-1])


def tree_to_list(program_tree):
    """
    Converts a program from a nested tree representation back to a
    dependency-ordered list.
    """

    def _count_nodes(node):
        """Helper to count total nodes to pre-allocate list."""
        return 1 + sum(_count_nodes(child) for child in node["inputs"])

    def _build_list_recursive(node, current_index, program_list):
        """
        Recursive helper to populate the list backward from the root.
        Returns the index of the *next* available slot (i.e., the one
        just before the children of this node).
        """
        program_list[current_index] = {
            "function": node["function"],
            "value_inputs": list(node["value_inputs"]),
            "inputs": [],
        }
        child_index = current_index - 1
        for child_node in reversed(node["inputs"]):
            # Add the *final* index of the child to the parent's input list
            program_list[current_index]["inputs"].insert(0, child_index)
            # Recursively build the child subtree, which returns the next
            # available index *before* that child's subtree
            child_index = _build_list_recursive(child_node, child_index, program_list)
        return child_index

    if not program_tree:
        return []

    num_nodes = _count_nodes(program_tree)
    output_list = [None] * num_nodes
    _build_list_recursive(program_tree, num_nodes - 1, output_list)
    return output_list


# --- Prefix Conversions ---


def tree_to_prefix(program_tree):
    """Converts a program tree to a prefix-ordered list."""
    prefix_list = []

    def _traverse(node):
        """Pre-order traversal."""
        prefix_list.append(
            {
                "function": node["function"],
                "value_inputs": list(node["value_inputs"]),
            }
        )
        for child in node["inputs"]:
            _traverse(child)

    if program_tree:
        _traverse(program_tree)
    return prefix_list


def prefix_to_tree(program_prefix):
    """Converts a prefix-ordered list into a program tree."""
    if not program_prefix:
        return None

    # Use a deque for efficient popping from the left
    prefix_queue = deque(program_prefix)

    def _build_from_prefix():
        """Recursive helper to build tree from prefix queue."""
        # Get the next function in the sequence
        func_node_data = prefix_queue.popleft()

        # Build the current node
        new_node = {
            "function": func_node_data["function"],
            "value_inputs": list(func_node_data["value_inputs"]),
            "inputs": [],
        }

        # Recursively build children based on arity
        num_inputs = get_num_inputs(new_node)
        for _ in range(num_inputs):
            new_node["inputs"].append(_build_from_prefix())

        return new_node

    return _build_from_prefix()


# --- Postfix Conversions ---


def tree_to_postfix(program_tree):
    """Converts a program tree to a postfix-ordered list."""
    postfix_list = []

    def _traverse(node):
        """Post-order traversal."""
        for child in node["inputs"]:
            _traverse(child)
        postfix_list.append(
            {
                "function": node["function"],
                "value_inputs": list(node["value_inputs"]),
            }
        )

    if program_tree:
        _traverse(program_tree)
    return postfix_list


def postfix_to_tree(program_postfix):
    """Converts a postfix-ordered list into a program tree."""
    if not program_postfix:
        return None

    # We can just use a list since pop() from the end is efficient
    postfix_stack = list(program_postfix)

    def _build_from_postfix():
        """Recursive helper to build tree from postfix stack."""
        # Get the next function (from the end)
        func_node_data = postfix_stack.pop()

        # Build the current node
        new_node = {
            "function": func_node_data["function"],
            "value_inputs": list(func_node_data["value_inputs"]),
            "inputs": [],
        }

        # Recursively build children
        num_inputs = get_num_inputs(new_node)
        # Build children in reverse order (since we're popping)
        for _ in range(num_inputs):
            # Insert at the beginning to maintain correct child order
            new_node["inputs"].insert(0, _build_from_postfix())

        return new_node

    return _build_from_postfix()


# --- Composite Conversions ---


def list_to_prefix(program_list):
    """Convenience function: list -> tree -> prefix"""
    return tree_to_prefix(list_to_tree(program_list))


def prefix_to_list(program_prefix):
    """Convenience function: prefix -> tree -> list"""
    return tree_to_list(prefix_to_tree(program_prefix))


def list_to_postfix(program_list):
    """Convenience function: list -> tree -> postfix"""
    return tree_to_postfix(list_to_tree(program_list))


def postfix_to_list(program_postfix):
    """Convenience function: postfix -> tree -> list"""
    return tree_to_list(postfix_to_tree(program_postfix))


# --- Validation ---


def is_chain(program_list):
    """
    Checks if a program list is a "chain", meaning no node has
    more than one child (i.e., no branching).
    """
    visited_indices = set()
    if not program_list:
        return True

    current_index = len(program_list) - 1
    while current_index is not None:
        if current_index in visited_indices:
            # This would indicate a cycle, which shouldn't happen
            return False
        visited_indices.add(current_index)

        node = program_list[current_index]
        inputs = node.get("inputs", [])

        if len(inputs) > 1:
            # A node has more than one input, so it's not a chain
            return False
        elif len(inputs) == 1:
            # Follow the chain
            current_index = inputs[0]
        else:
            # Reached the start of the chain (e.g., 'scene' node)
            current_index = None

    # If we visited all nodes, it's a valid chain
    return len(visited_indices) == len(program_list)
