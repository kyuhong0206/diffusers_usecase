import argparse
from diffusers import StableDiffusionPipeline

def convert_model_to_diffusers(input_path, output_path):
    pipe = StableDiffusionPipeline.from_single_file(input_path)
    pipe.save_pretrained(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type = str)
    parser.add_argument('--output_path', type = str)
    args = parser.parse_args()
    convert_model_to_diffusers(args.input_path, args.output_path)
