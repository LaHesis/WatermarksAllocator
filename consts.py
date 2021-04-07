import os
from time import time

# LITERALS
PROGRAM_NAME = 'Watermarks allocator'
PROGRAM_DESCRIPTION = """
Allows to process several images and automatically choose suitable
allocations for watermarks. See readme.md for more info.
"""

# Delta which gets added to X and Y coords during finding of areas suitable
# for watermark.
STEP = 16
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 100
SHOW_WMARK_AREA_OUTLINES = False

# LITERALS RELATED TO IMAGE AREA SUITABILITY CHECKS
# If an image area contains pixels with a lower value in amount of greater than
# DARK_PXLS_ALLOWED_AMOUNT, no watermark would be placed there.
# Should be between 1 and 255.
DARK_PXLS_THRESHOLD = 80
DARK_PXLS_ALLOWED_AMOUNT = 20
EDGE_PXLS_ALLOWED_AMOUNT = 10

ACCEPTABLE_FILE_EXTENSIONS = ('png', 'jpg', 'jpeg')

# Generated constants.
PROGRAM_DESCRIPTION = PROGRAM_NAME + PROGRAM_DESCRIPTION
PROGRAM_DIR = os.path.abspath(os.path.dirname(__file__))
START_TIME = time()
