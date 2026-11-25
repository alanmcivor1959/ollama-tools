import json
import label_classes
import re
from typing import Literal
from pydantic import Field, create_model

def parse_model_response(response_text, img_w, img_h, classes_list, verbose=False):
    """
    Parses the JSON string response from Qwen3-VL and extracts the details.
    """

    #
    # Here are some examples of observed output that this parser does not yet handle:
    #
    #[
    #  {"person walking, running, or standing": "[764, 918, 835, 997]"}
    #] 
    #(Currently this gets rejected)
    #
    #[
    #  {"bbox_2d": [634, 80, 671, 252], "label": "person standing"}
    #]
    #(Currently this gets rejected)
    #
    
    cleaned_text = ""
    try:
        # Regex to find content inside ```json ... ``` or just ``` ... ```
        # It searches for the pattern across multiple lines (re.DOTALL)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
        
        if json_match:
            # Extract the raw string inside the code blocks
            cleaned_text = json_match.group(1).strip()
        else:
            # Fallback: Check if the text actually looks like a raw JSON array/object
            stripped_response = response_text.strip()
            if stripped_response.startswith(('[', '{')):
                cleaned_text = stripped_response
            else:
                # No backticks and no JSON structural markers. 
                # The model returned a pure text statement like "There is no object...".
                if verbose:
                    print("   [-] Pure text response received (No objects found).")
                return []

        # Parse the JSON string. With a format schema (see get_format()) the
        # response is an object {"objects": [...]}; without one it is a raw list.
        data = json.loads(cleaned_text)
        if isinstance(data, dict):
            data = data.get("objects")

        # Guard clause: Check if the list is empty ([])
        if not data or not isinstance(data, list):
            if verbose:
                print("   [-] No objects detected in this frame (Empty list returned).")
            return []
        
        if verbose:
            print("Parser:")
            print(f"--- Extracted {len(data)} detected items ---")
        
        bboxes = []

        # Iterate and extract the details
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if not label:
                continue
            bbox = item.get("bbox_2d")
            if not bbox or len(bbox) != 4:
                continue
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox

            item = label_classes.get_entry_from_prompt(classes_list, label)
            if item is None:
                continue
            code = item["code"]
            name = item["name"]
            
            # Denormalize from 1000-grid to actual image plane coordinates
            xmin = ( float(xmin_norm) / 1000.0) * float(img_w)
            ymin = ( float(ymin_norm) / 1000.0) * float(img_h)
            xmax = ( float(xmax_norm) / 1000.0) * float(img_w)
            ymax = ( float(ymax_norm) / 1000.0) * float(img_h)

            bbox = [ xmin, ymin, xmax, ymax ]
            
            if verbose:
                print(f"Item {index}:")
                print(f"  • Label: {label}")
                print(f"  • Code: {code}")
                print(f"  • Name: {name}")
                print(f"  • Bounding Box: {bbox}")

            bboxes.append( [ index, 0, 0.0, bbox, code ] )

        if verbose:
            print("")
            
        if len(bboxes) != len(data):
            if verbose:
                print(f"Only {len(bboxes)} bboxes extracted from {len(data)} items")
            return None

        return bboxes

    except json.JSONDecodeError as e:
        print(f"Error parsing response: The output was not valid JSON. Details: {e}")
        print("Raw response was:", response_text)
        return None


def get_class_prompts(classes_list):
    """Returns the prompt fields of the classes list, in prompt_order"""
    return [item["prompt"] for item in sorted(classes_list, key=lambda x: x["prompt_order"])]


def get_format(classes_list):
    """
    Returns the JSON schema for Ollama's format parameter, built from the classes list.

    The response must be an object {"objects": [{"label": ..., "bbox_2d": [...]}, ...]}.
    label is restricted to the exact prompt strings of the classes, so it always matches
    an entry via label_classes.get_entry_from_prompt().
    """
    prompts = get_class_prompts(classes_list)
    # Prompts can contain commas, so quote each one in the description
    labels = "; ".join(f'"{p}"' for p in prompts)
    detected_object = create_model(
        "DetectedObject",
        label=(Literal[tuple(prompts)], Field(description=f"Object class, exactly one of: {labels}")),
        bbox_2d=(list[int], Field(
            description="2D bounding box [x1, y1, x2, y2] normalized from 0 to 1000",
            min_length=4,
            max_length=4,
        )),
    )
    frame_detections = create_model("FrameDetections", objects=(list[detected_object], ...))
    return frame_detections.model_json_schema()


def get_prompt(classes_list):
    """Calculates prompt from classes list. The requested output matches get_format()."""
    prompts = get_class_prompts(classes_list)
    prompt = f"Detect all objects from any of the following {len(prompts)} classes:\n"
    for index, cprompt in enumerate(prompts, start=1):
        prompt += f"{index}. {cprompt}\n"
    prompt += (
        'Respond with a JSON object with a key "objects" containing a list of the detected objects. '
        'Each object has a "label", which must be exactly one of the class descriptions above, '
        'and a "bbox_2d" giving the bounding box [x1, y1, x2, y2] normalized from 0 to 1000. '
        'If no objects are detected, "objects" is an empty list.'
    )
    return prompt


    
