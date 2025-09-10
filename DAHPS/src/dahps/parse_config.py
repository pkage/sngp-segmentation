from copy import deepcopy as copy
import json


def bfs_paths(config, partial):
    assert isinstance(config, dict) or config is None, (
        f"config must be a dictionary or None, found: {type(config)}"
    )

    if config is None:
        return partial

    assert config["values"], "values list cannot be empty"
    assert isinstance(config["key"], str), "keys for parameters must be string names"
    for k in config.keys():
        if k not in ["key", "values", "default"]:
            assert k in config["values"], (
                "all specified cases must be members of the value set"
            )

    new = []
    key = config["key"]
    for value in config["values"]:
        if value in config:
            tmp = []
            for p in partial:
                tmp_p = copy(p)
                assert key not in p, (
                    "found duplicate key while parsing config parameters"
                )
                tmp_p[key] = value
                tmp.append(tmp_p)

            new += bfs_paths(config[value], tmp)
        else:
            tmp = []
            for p in partial:
                tmp_p = copy(p)
                assert key not in p, (
                    "found duplicate key while parsing config parameters"
                )
                tmp_p[key] = value
                tmp.append(tmp_p)

            new += bfs_paths(config["default"], tmp)

    return new


def parse_parameter_config(config):
    # returns a list or set of parameter dictionaries
    root = config["root"]
    paths = bfs_paths(root, [{}])

    print(
        f"found {len(paths)} specified configurations of which {len(set([tuple(sorted([(k, v) for k, v in p.items()], key=lambda k: k[0])) for p in paths]))} are unique"
    )

    return paths
