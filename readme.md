# Watermarks allocator

The script inserts watermark image into each image existing in input directory. Watermarks are inserted at suitable areas of an image. By default image areas are considered suitable if they have no sharp transitions (edges). I.e. `consider_edges` execution argument is set to `1` by default. OCR (Tesseract-OCR) and dark pixels presence check can be turned on to not allocate watermarks if there are detected text or dark pixels (in amount of greater than `DARK_PXLS_ALLOWED_AMOUNT` literal) in an image area respectively. Therefore, it is not recommended to turn on the dark pixels presence check for images with black or dark background. To turn on Tesseract-OCR, specify `--use_tesseract 1` script execution argument. To turn on dark pixels presence check, specify `--consider_dark_pixels 1` script execution argument. Use of Tesseract-OCR increases script execution time but can reduce amount of watermark wrong allocations, especially when text color doesn't much contrast with image background.

There are some script execution arguments. Use `-h` execution argument to see their description or look at _main.py_ itself. Watermark image is checked for existence. In the cases when watermark image doesn't exist, input directory doesn't exist or input directory has no images, the script prints an error message and stops.

`used_areas_percentage` execution argument specifies how many watermarks are inserted onto all suitable areas. Default coverage is 70. In the case when `used_areas_percentage` or amount of founded suitable areas are too low, and the amount of watermarks to insert is 0, then 1 watermark is inserted (if there is at least 1 suitable area). When there are no suitable areas at all for an input image, corresponding warning is printed.

Demo with default execution arguments:

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/default parameters demo.png>)

Inverted demo with default execution arguments (pay attention that `consider_dark_pixels` script execution argument should be not specified or be set to 0 for such images having dark background and light font color):

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/demo with inverted colors and default parameters.png>)

Demo with `--min_scale 0.1 --used_areas_percentage 100`:

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/demo with -ms 0.1 -uap 100.png>)

Used for photo with default execution arguments (https://www.publicdomainpictures.net/en/view-image.php?image=305171&picture=yarrow-blossom):

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/schafgarbe-blute default parameters demo.jpg>)

Used for photo with `used_areas_percentage` script execution argument set to 15 (https://www.publicdomainpictures.net/en/view-image.php?image=305171&picture=yarrow-blossom):

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/schafgarbe-blute used_areas_percentage is 15.jpg>)

Photo processed with `used_areas_percentage` script execution argument is set to 100 and no suitability check is turned on (`--consider_edges 0 --consider_dark_pixels 0`) (https://www.publicdomainpictures.net/en/view-image.php?image=305171&picture=yarrow-blossom):

![Demo with default execution arguments](<https://github.com/LaHesis/WatermarksAllocator/raw/master/demo/schafgarbe-blute used_areas_percentage is 100 and no suitability check used.jpg>)

The following packages are used:
- argparse 1.4.0 (by some reason it doesn't show up when using `pip freeze`);
- numpy 1.19.4;
- opencv-python 4.4.0.46;
- Pillow 8.0.1;
- progress 1.5;
- pytesseract 0.3.6.

The script was tested on Windows 10 64-b.
