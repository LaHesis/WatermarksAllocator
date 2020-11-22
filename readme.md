#Watermarks allocator

The script inserts watermark image placed at WM_PATH into each image
placed at INPUT_IMGS_DIR_NAME. Watermarks are inserted at suitable areas
of an image. By default image areas are considered suitable if they have no
sharp transitions (edges).

Results are placed at OUTPUT_IMGS_DIR_NAME directory. Script can use Tesseract
to identify image areas having text and can use OpenCV for 1 default check and
1 additional check which are performed when some area has no text detected by
Tesseract.

The 2 checks are:
1. Does current image area have pixels darker than DARK_PXLS_THRESHOLD
in an amount of greater than DARK_PXLS_ALLOWED_AMOUNT.
2. Does current image area have edge pixels in an amount of greater than
DARK_PXLS_ALLOWED_AMOUNT (used by default).

If any of the checks gives True for some area, watermark is not placed. By
default only the second check is used, i. e. Tesseract and the first check
are not used. It is not recommended to use the first check for images with
black or dark background.

COVERAGE means how many watermarks are inserted onto all suitable areas. In
the case when COVERAGE or suitable areas amount are too low and the amount
of watermarks to insert is 0, then 1 watermark is inserted (if there is at least
1 suitable area). When there are no suitable areas at all for an input image,
corresponding warning is printed.