from ollama import Client
import argparse
import qwen3vl_support as q3
import label_classes
import bbox_io as bbio
from PIL import Image
import tempfile
import cv2
import os
import sys
import time

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
    parser.add_argument("--restart", action='store_true',
                        help="Resume an interrupted run: append to --output, starting after the last frame in it")
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

    out_mode = "w"
    if args.restart:
        out_mode = "a"
        start_frame = 1
        if os.path.exists(args.output):
            existing = bbio.read(args.output, img_w, img_h)
            if existing:
                # Continue after the last frame and bbox id already written
                start_frame = max(b[1] for b in existing) + 1
                bboxid = max(b[0] for b in existing)
        # Skip frames already processed (grab() avoids decoding them)
        for _ in range(start_frame - 1):
            if not cap.grab():
                print(f"Error: video has fewer than {start_frame} frames.")
                sys.exit(1)
        print(f"Restarting at frame {start_frame}, continuing from bbox id {bboxid}")

    total_duration = 0
    load_duration = 0
    prompt_eval_count = 0
    prompt_eval_duration = 0
    eval_count = 0
    eval_duration = 0

    num_unparsed = 0
    num_frames = 0

    start_time = time.perf_counter()

    with open(args.output, out_mode, encoding="utf-8") as out_f:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            num_frames += 1
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
                out_f.write(bbio.format_bbox(bbox, img_w, img_h))
            out_f.flush()

    cap.release()

    elapsed = time.perf_counter() - start_time

    if num_frames == 0:
        print("No frames processed.")
        return

    total_duration /= num_frames
    load_duration /= num_frames
    prompt_eval_count /= num_frames
    prompt_eval_duration /= num_frames
    eval_count /= num_frames
    eval_duration /= num_frames
    print(f"Ave(total_duration):       {total_duration*1e-9}")
    print(f"Ave(load_duration):        {load_duration*1e-9}")
    print(f"Ave(prompt_eval_count):    {prompt_eval_count}")
    print(f"Ave(prompt_eval_duration): {prompt_eval_duration*1e-9}")
    print(f"Ave(eval_count):           {eval_count}")
    print(f"Ave(eval_duration):        {eval_duration*1e-9}")

    print(f"Total elapsed time:        {elapsed:.3f} s")
    print(f"Elapsed time per frame:    {elapsed/num_frames:.3f} s")

    print(f"Number frame responses unparsed: {num_unparsed} ({100*num_unparsed/num_frames}%)")

if __name__ == "__main__":
    main()
