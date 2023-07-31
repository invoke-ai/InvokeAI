from collections import OrderedDict
from typing import List


def unique_list(any_list: List) -> List:
    """Takes in a list and removes duplicates while maintaining item order"""
    return list(OrderedDict.fromkeys(any_list))
