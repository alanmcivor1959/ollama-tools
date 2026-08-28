import hashlib
import os
import json


def read_label_classes(fname):
    """Reads label classes from json file"""
    try:
        with open(fname, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Syntax Error (check for trailing commas!): {e}")
    except FileNotFoundError:
        print("The specified file was not found.")


def get_class_colour(label_name):
    """
    Generates a deterministic, high-visibility RGB colour based on the label text string.
    Ensures the same object class always gets the same colour.
    """
    # Create a stable hash of the label text
    hash_object = hashlib.md5(label_name.lower().strip().encode())
    hex_dig = hash_object.hexdigest()

    # Extract RGB values using chunks of the hash
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)

    # Maximize brightness/saturation for visual clarity against varied image backgrounds
    max_val = max(r, g, b, 1)
    r = int((r / max_val) * 200) + 55
    g = int((g / max_val) * 200) + 55
    b = int((b / max_val) * 200) + 55

    return (r, g, b)


def get_entry_from_prompt(classes_list, prompt):
    """
    Get the entry on the classes_list that has the given prompt.
    If no exact match, it searches for item whose prompt entry
    starts with the given prompt
    Returns None if prompt invalid
    """
    for item in classes_list:
        if item.get("prompt") == prompt:
            return item
    n = 0
    iii = None
    for item in classes_list:
        if item.get("prompt").startswith(prompt):
             n += 1
             iii = item
    if n == 1:
        return iii
    return None
