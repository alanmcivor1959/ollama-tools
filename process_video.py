from ollama import Client
import argparse
import qwen3vl_support as q3
import label_classes
import bbox_io as bbio
from PIL import Image
import tempfile
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Process image with Ollama")
    parser.add_argument("--video", type=str, required=True, help="Path to local video file")
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument("--output", type=str, required=True, help="Filename to save the bbox to")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b", help="Ollama model id")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="Address of Ollama server")
#    parser.add_argument("--batch", type=int, default=16, help="Number of frames per batch")
    parser.add_argument("--verbose", action='store_true', help="Show raw output too")
    return parser.parse_args()


def main():
    args = parse_args()

    client = Client(host=f"http://{args.server}:11434")

    model = args.model
    verbose = args.verbose
    classes_list = label_classes.read_label_classes(args.classes)

    prompt = label_classes.get_prompt(classes_list)
    if verbose:
        print("Prompt:")
        print( prompt )
        print("")

#    batch_size = args.batch

    bboxes = []
    bboxid = 0

#    frame_no = []
#    frame_ts = []

    # 2. Process video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        fts = 0.0               # stub

        if verbose:
            print("")
            print("")
            print(f"Processing frame {fno}")
            print("")

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file.close()

        cv2.imwrite(temp_file.name, frame)
        
        response = client.chat(
            model = model,
            messages = [
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [ temp_file.name ]
                }
            ]
        )

        os.unlink(temp_file.name)
        
        if ( verbose ):
            print("Raw response:")
            print(response['message']['content'])
            print("")

        fbboxes = []

        if ( model == "qwen3-vl:8b" ):
            fbboxes = q3.parse_model_response(response['message']['content'], img_w, img_h, classes_list, verbose)

        if ( verbose ):
            print( "BBoxes:")
            print( fbboxes )
            print("")

        for bbox in fbboxes:
            bboxid += 1
            bbox[0] = bboxid
            bbox[1] = fno
            bbox[2] = fts
            bboxes.append(bbox)

        # TEMP
        bbio.write("output.bbox", bboxes, img_w, img_h)

    cap.release()

    bbio.write(args.output, bboxes, img_w, img_h)


if __name__ == "__main__":
    main()
