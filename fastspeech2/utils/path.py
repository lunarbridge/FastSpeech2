import re
from typing import List


def get_rest_path_from(exclude_paths: List,
                       search_path_pattern: str = None,
                       search_path_filter_regex: re.Pattern = None) -> List:
    import glob

    rest_path = glob.glob(search_path_pattern)
    if search_path_filter_regex:
        rest_path = [path for path in rest_path if not re.search(search_path_filter_regex, path)]
    for exclude_path in exclude_paths:
        rest_path = [path for path in rest_path if exclude_path not in path]

    return rest_path
