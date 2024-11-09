import json
import argparse

from pipeline.write_script import write_script
from pipeline.script_to_speech import script_to_speech
from pipeline.utils import upload_to_gemini, wait_for_files_active

def parse_args():
    """
    Parse the arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf-path', type=str, help='Path to the PDF file')
    parser.add_argument('--save-intermediate-outputs', action='store_true', help='Save intermediate outputs')
    return parser.parse_args()

def main(parsed_args):
    """
    parsed_args: the arguments from the command line
    """
    pdf_in_gemini = upload_to_gemini(parsed_args.pdf_path)
    wait_for_files_active([pdf_in_gemini])

    script = write_script(pdf_in_gemini)
    if parsed_args.save_intermediate_outputs:
        with open("raw_script.json", "w", encoding="utf-8") as f:
            json.dump(script, f)

    # script = refine_script(script)
    # if parsed_args.save_intermediate_outputs:
    #     with open("refined_script.json", "w", encoding="utf-8") as f:
    #         json.dump(script, f)
    
    audio = script_to_speech(script)
    audio.export("./_podcast.mp3", format="mp3", bitrate="192k", parameters=["-q:a", "0"])

if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
