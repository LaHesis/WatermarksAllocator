import argparse

import consts as c


def _create_arg_parser():
    arg_parser = argparse.ArgumentParser(description=c.PROGRAM_DESCRIPTION)
    arg_parser.add_argument(
        '-w', '--wm_path', required=True,
        help='Path to watermark image (can have transparency).'
    )
    arg_parser.add_argument(
        '-i', '--input_dir', required=True,
        help='Path to the input directory of images.'
    )
    arg_parser.add_argument(
        '-o', '--output_dir', required=True,
        help='Path to the output directory.'
    )
    arg_parser.add_argument(
        '-ms', '--min_scale', type=float, default=.5,
        help='Minimal scale which watermark will be resized to. Can be between 0.1 and 1.0, inclusively. Default value is 0.5.'
    )
    arg_parser.add_argument(
        '-ut', '--use_tesseract', type=bool, default=0,
        help='0 means the script will not use OCR (Tesseract-OCR) to not allocate watermarks onto text, 1 means the opposite. Use of Tesseract increases execution time but can improve accuracy especially when text color doesn\'t much contrast with background. Default value is 0.'
    )
    arg_parser.add_argument(
        '-t', '--tesseract_path', default='C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
        help='Full path to Tesseract-OCR. Default value is "C:\\Program Files\\Tesseract-OCR\\tesseract.exe". Is used only if --use_tesseract is 1 or True.'
    )
    arg_parser.add_argument(
        '-cdp', '--consider_dark_pixels', type=int, default=0,
        help='1 means the script will not allocate watermarks on areas having dark pixels, 0 means the script will ignore the factor during area suitability check. Default value is 0.'
    )
    arg_parser.add_argument(
        '-ce', '--consider_edges', type=int, default=1,
        help='1 means the script will not allocate watermarks on areas having edges (sharp transition of colors), 0 means the script will ignore the factor during area suitability check. Default value is 1.'
    )
    arg_parser.add_argument(
        '-uap', '--used_areas_percentage', type=int, default=70,
        help='Percentage of image suitable areas where watermarks will be inserted to.'
    )
    return arg_parser


_arg_parser = _create_arg_parser()
# Read arguments values.
args = vars(_arg_parser.parse_args())
wm_path = args['wm_path']
input_dir = args['input_dir']
output_dir = args['output_dir']
min_wm_scale: float = args['min_scale']
# Whether to use Tesseract.
use_OCR = args['use_tesseract']
tesseract_path = args['tesseract_path']
consider_dark_pxls = args['consider_dark_pixels']
consider_edges = args['consider_edges']
used_areas_percentage = args['used_areas_percentage']
