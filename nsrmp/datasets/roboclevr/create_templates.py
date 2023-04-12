import os
import json
from copy import deepcopy
from collections import OrderedDict

TEMPLATES_DIR = "../../../data_generation/panda/construct/templates"
OUTPUT_FILE = "program_templates_new.json"


def compress_filters(orig, k=0):
    if k + 1 >= len(orig["nodes"]):
        return [orig]
    if (
        "filter" in orig["nodes"][k]["type"]
        and "filter" in orig["nodes"][k + 1]["type"]
    ):
        removed = deepcopy(orig)
        removed["nodes"] = [node for i, node in enumerate(removed["nodes"]) if i != k]
        for i in range(k, len(removed["nodes"])):
            removed["nodes"][i]["inputs"] = [
                inp - 1 if inp >= k else inp for inp in removed["nodes"][i]["inputs"]
            ]
        return compress_filters(orig, k + 1) + compress_filters(removed, k)
    return compress_filters(orig, k + 1)


def convert_template(orig):
    ls = []
    for node in orig["nodes"]:
        if node["type"] == "idle":
            ls.append(OrderedDict([("op", "idle"), ("inputs", [])]))
        elif node["type"] == "scene":
            ls.append(OrderedDict([("op", "scene"), ("inputs", [])]))
        elif "filter" in node["type"]:
            assert len(node["inputs"]) == 1, node
            ls.append(
                OrderedDict(
                    [
                        ("op", "filter"),
                        ("inputs", [ls[node["inputs"][0]]]),
                        ("attribute_concept_idx", None),
                        ("attribute_concept_values", None),
                    ]
                )
            )
        elif node["type"] == "relate":
            assert len(node["inputs"]) == 1
            ls.append(
                OrderedDict(
                    [
                        ("op", "relate"),
                        ("inputs", [ls[node["inputs"][0]]]),
                        ("relational_concept_idx", None),
                        ("relational_concept_values", None),
                    ]
                )
            )
        elif node["type"] == "unique":
            assert len(node["inputs"]) == 1
            ls.append(ls[node["inputs"][0]])
        elif node["type"] == "move":
            assert len(node["inputs"]) == 3
            ls.append(
                OrderedDict(
                    [
                        ("op", "move"),
                        ("inputs", [ls[inp] for inp in node["inputs"]]),
                        ("action_concept_idx", None),
                        ("action_concept_values", None),
                    ]
                )
            )

    return ls[-1]


def preorder(node):
    return node["op"] + "(" + ",".join([preorder(c) for c in node["inputs"]]) + ")"


def scan_input_templates():
    prog_templates = {}
    for template_f in next(os.walk(TEMPLATES_DIR))[2]:
        print(template_f)
        with open(os.path.join(TEMPLATES_DIR, template_f), "r") as f:
            templates = json.load(f)
        for template in templates:
            variations = compress_filters(template)
            print(len(variations))
            for variation in variations:
                converted_template = convert_template(variation)
                prog_templates[preorder(converted_template)] = converted_template
    with open(OUTPUT_FILE, "w") as f:
        json.dump(prog_templates, f, sort_keys=False)


scan_input_templates()
