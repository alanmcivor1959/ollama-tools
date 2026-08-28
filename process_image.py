from ollama import Client
import argparse
import qwen3vl_support as q3
import label_classes
import bbox_io as bbio
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Process image with Ollama")
    parser.add_argument("--image", type=str, required=True, help="Path to local image file")
    parser.add_argument("--classes", type=str, required=True, help="Path to json file defining classes")
    parser.add_argument("--output", type=str, required=True, help="Filename to save the bbox to")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b", help="Ollama model id")
    parser.add_argument("--server", type=str, default="127.0.0.1", help="Address of Ollama server")
#    parser.add_argument("--batch", type=int, default=16, help="Number of frames per batch")
#    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens in response")
    parser.add_argument("--verbose", action='store_true', help="Show raw output too")
    return parser.parse_args()


def main():
    args = parse_args()

    client = Client(host=f"http://{args.server}:11434")

    model = args.model
    verbose = args.verbose
    classes_list = label_classes.read_label_classes(args.classes)

    image = Image.open(args.image)
    img_w, img_h = image.size
    if verbose:
        print("Image Dimensions")
        print(f"\tWidth = {img_w}")
        print(f"\tHeight = {img_h}")
        print("")

    prompt = q3.get_prompt(classes_list)
    if verbose:
        print("Prompt:")
        print( prompt )
        print("")

    response = client.chat(
        model = model,
        messages = [
            {
                'role': 'user',
                'content': prompt,
                'images': [ args.image ]
            }
        ]
    )

    if ( verbose ):
        print("Raw response:")
        print(response['message']['content'])
        print("")

    bboxes = []

    if ( model == "qwen3-vl:8b" ):
        bboxes = q3.parse_model_response(response['message']['content'], img_w, img_h, classes_list, verbose)

    if ( verbose ):
        print( "BBoxes:")
        print( bboxes )
        print("")

    bbio.write(args.output, bboxes, img_w, img_h)


if __name__ == "__main__":
    main()
