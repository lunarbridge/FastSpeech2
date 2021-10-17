import os
import re
from typing import Dict, List
import re

# from fastspeech2.utils import data_layout


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


# class PathNode:
#     def __init__(self, name, contents: List = None, display: str = None) -> None:
#         self.name = name
#         self.contents = contents

#         if not display:
#             self.display = name
#         else:
#             self.display = display


# def parse_data_layout(layout: PathNode) -> Dict:
#     paths = dict()
#     traverse_stack = []

#     def _register_path(path_key, path_items: List):
#         if path_key in paths:
#             raise ValueError(f'path key "{path_key}" duplicates')

#         paths[path_key] = os.path.sep.join(path_items)


#     def _traverse(d: PathNode):
#         if not d.contents:
#             _register_path(path_key=d.name, path_items=traverse_stack + [d.display])
#             traverse_stack.append(d.display)
#             return

#         traverse_stack.append(d.display)
#         _register_path(path_key=d.name, path_items=traverse_stack)

#         for content in d.contents:
#             if isinstance(content, PathNode):
#                 _traverse(content)

#                 traverse_stack.pop()

#     _traverse(layout)

#     return paths


def get_data_path_key(current_data_path_abspath: str) -> str:
    data_path_key_regex = '\d{8}-\d{6}$|\d{8}-\d{6}-intermediate$'

    path_key_match = re.search(data_path_key_regex, current_data_path_abspath)
    if not path_key_match:
        raise AttributeError(f'Invalid data path key.\n'
                             f'current_data_path: {current_data_path_abspath}\n'
                             f'key match pattern: {data_path_key_regex}')

    return path_key_match.group()


# def get_fs2_paths(current_data_path: str = None) -> Dict:
#     from fastspeech2.utils.data_layout import fs2_data_layout

#     data_layout = parse_data_layout(fs2_data_layout)
#     if current_data_path:
#         current_data_path_key = get_data_path_key(current_data_path)
#         data_layout = {k: re.sub('current_data_template', current_data_path_key, path) for k, path in data_layout.items()}

#     return data_layout
