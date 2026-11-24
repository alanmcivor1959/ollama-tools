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
    parser.add_argument("--temperature", type=float, help="Creativity of the model (0.0 to 1.0)")
    parser.add_argument("--num_ctx", type=int, default=4096, help="Context length")
    parser.add_argument("--num_predict", type=int, default=-1, help="Maximum number of output tokens")
    parser.add_argument("--verbose", action='store_true', help="Show raw output too")
    return parser.parse_args()


def main():
    args = parse_args()

    client = Client(host=f"http://{args.server}:11434")

    model = args.model
    verbose = args.verbose
    classes_list = label_classes.read_label_classes(args.classes)

    ollama_options = {}
    if args.temperature is not None:
        ollama_options['temperature'] = args.temperature
    ollama_options['num_ctx'] = args.num_ctx;
    ollama_options['num_predict'] = args.num_predict;

    prompt = q3.get_prompt(classes_list)
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

    total_duration = 0
    load_duration = 0
    prompt_eval_count = 0
    prompt_eval_duration = 0
    eval_count = 0
    eval_duration = 0

    num_unparsed = 0

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
        
        response = client.generate(
            model = model,
            prompt = prompt,
            images = [ temp_file.name ],
            options = ollama_options
        )

        os.unlink(temp_file.name)
        
        if ( verbose ):
            print("Raw response:")
            print(response['response'])
            print("")

        if ( response['done'] != True ) or ( response['done_reason'] != "stop" ):
            print(f"Incorrect stop criteria {response['done']} {response['done_reason']}")
            sys.exit(1)

        total_duration += response['total_duration']
        load_duration += response['load_duration']
        prompt_eval_count += response['prompt_eval_count']
        prompt_eval_duration += response['prompt_eval_duration']
        eval_count += response['eval_count']
        eval_duration += response['eval_duration']

        fbboxes = []

        if ( model == "qwen3-vl:8b" ):
            fbboxes = q3.parse_model_response(response['response'], img_w, img_h, classes_list, verbose)

        if fbboxes is None:
            num_unparsed += 1;
            continue
            
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

    cap.release()

    bbio.write(args.output, bboxes, img_w, img_h)

    total_duration /= fno
    load_duration /= fno
    prompt_eval_count /= fno
    prompt_eval_duration /= fno
    eval_count /= fno
    eval_duration /= fno
    print(f"Ave(total_duration):       {total_duration*1e-9}")
    print(f"Ave(load_duration):        {load_duration*1e-9}")
    print(f"Ave(prompt_eval_count):    {prompt_eval_count}")
    print(f"Ave(prompt_eval_duration): {prompt_eval_duration*1e-9}")
    print(f"Ave(eval_count):           {eval_count}")
    print(f"Ave(eval_duration):        {eval_duration*1e-9}")

    print(f"Number frame responses unparsed: {num_unparsed} ({100*num_unparsed/fno}%)")

if __name__ == "__main__":
    main()
