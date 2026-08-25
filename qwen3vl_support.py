import json
import label_classes

def parse_model_response(response_text, img_w, img_h, classes_list, verbose=False):
    """
    Parses the JSON string response from Qwen3-VL and extracts the details.
    """
    try:
        # If the response has leading/trailing markdown code blocks (```json ... ```), clean them up
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            # Remove the opening fence line and closing fence line
            lines = cleaned_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned_text = "\n".join(lines).strip()

        # Parse the JSON string into a Python list
        data = json.loads(cleaned_text)
        
        if verbose:
            print("Parser:")
            print(f"--- Extracted {len(data)} detected items ---")
        
        bboxes = []

        # Iterate and extract the details
        for index, item in enumerate(data, start=1):
            label = item.get("label", "Unknown")
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = item.get("bbox_2d", [])

            item = label_classes.get_entry_from_prompt(classes_list, label)
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

