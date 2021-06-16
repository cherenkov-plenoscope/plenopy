import json


def read_trigger_response_from_path(path):
    with open(path, "rt") as fin:
        trigger_response = json.loads(fin.read())
    return trigger_response


def write_trigger_response_to_path(trigger_response, path):
    with open(path, "wt") as fout:
        fout.write(json.dumps(trigger_response, indent=4))
