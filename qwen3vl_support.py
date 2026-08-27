import json
import label_classes
import re

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

        # Parse the JSON string into a Python list
        data = json.loads(cleaned_text)

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
            label = item.get("label", "Unknown")
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
            
        return bboxes

    except json.JSONDecodeError as e:
        print(f"Error parsing response: The output was not valid JSON. Details: {e}")
        print("Raw response was:", response_text)
        return None

