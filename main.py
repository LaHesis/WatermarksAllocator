import os
from random import shuffle
from time import time

import cv2
import numpy as np
import pytesseract
from progress.spinner import Spinner

import arguments as args
import consts as c


def whether_rectangles_overlap(r1_coords, r2_coords):
    if r1_coords[0] >= r2_coords[2] or r1_coords[2] <= r2_coords[0] \
            or r1_coords[3] <= r2_coords[1] or r1_coords[1] >= r2_coords[3]:
        return False
    return True


def get_scaled_watermarks(wmark_img, min_scale):
    scaled_wmark_imgs = {}
    for wmark_scale in scale_generator(min_scale):
        scaled_wmark_imgs[wmark_scale] = cv2.resize(
            wmark_img, (0, 0), fx=wmark_scale, fy=wmark_scale
        )
    return scaled_wmark_imgs


def scale_generator(min_scale):
    for s_i in range(10, int(min_scale * 10) - 1, -1):
        wmark_scale_to_yield = round(s_i / 10, 1)
        yield wmark_scale_to_yield


def find_suitable_areas(unsuitable_blocks, img_w, img_h, gr_img, edges_img):
    """Find areas to place watermarks having different size.

    Arguments:
        not_suitable_blocks {[type]} -- areas unsuitable for watermarks.
    """
    suitable_areas = []
    for wmark_scale in scale_generator(args.min_wm_scale):
        cur_scl_not_suitable_blocks = set(unsuitable_blocks)
        spinner = Spinner(
            f'Available space estimating for watermark scale {wmark_scale}...',
        )
        # Watermark current width and height in pixels.
        scaled_mark_width = round(wm_w * wmark_scale)
        scaled_mark_height = round(wm_h * wmark_scale)
        # The values are used to determine if at least one watermark can be
        # placed.
        marks_in_img_along_x = img_w // scaled_mark_width
        marks_in_img_along_y = img_h // scaled_mark_height
        if marks_in_img_along_x > 0 and marks_in_img_along_y > 0:
            y = 0
            while y + scaled_mark_height < img_h:
                x = 0
                # Filter out blocks above y.
                cur_scl_not_suitable_blocks = set(filter(
                    lambda bl: bl[3] >= y,
                    cur_scl_not_suitable_blocks
                ))
                while x + scaled_mark_width < img_w:
                    potential_mark_rect = (
                        x, y, x + scaled_mark_width,
                        y + scaled_mark_height
                    )
                    overlapped_blocks = list(filter(
                        lambda bl: whether_rectangles_overlap(bl, potential_mark_rect),
                        cur_scl_not_suitable_blocks
                    ))
                    if len(overlapped_blocks) == 0:
                        # ROI - region of interest, it's an input image
                        # fragment.
                        gr_roi = gr_img[
                            y:y + scaled_mark_height, x:x + scaled_mark_width
                        ].flatten()
                        edges_roi = edges_img[
                            y:y + scaled_mark_height, x:x + scaled_mark_width
                        ].flatten()
                        dark_pxls_amount = (gr_roi < c.DARK_PXLS_THRESHOLD).sum()
                        too_many_dark_pxls = dark_pxls_amount > c.DARK_PXLS_ALLOWED_AMOUNT
                        edge_pxls_amount = (edges_roi > 0).sum()
                        too_many_edges = edge_pxls_amount > c.EDGE_PXLS_ALLOWED_AMOUNT
                        if (not args.consider_dark_pxls or not too_many_dark_pxls) and (not args.consider_edges or not too_many_edges):
                            suitable_areas.append(
                                {
                                    'rect_coords': potential_mark_rect,
                                    'wmark_scale': wmark_scale
                                }
                            )
                            x += scaled_mark_width - c.STEP
                            cur_scl_not_suitable_blocks.add(
                                potential_mark_rect
                            )
                    x += c.STEP
                unsuitable_blocks |= cur_scl_not_suitable_blocks
                y += c.STEP
                spinner.next()
        spinner.finish()
    return suitable_areas


def insert_watermarks(input_img_path):
    print(f'"{input_img_path}" is being processed...')
    input_img_path = os.path.join(
        c.PROGRAM_DIR, args.input_dir, input_img_path
    )
    input_img = cv2.imread(input_img_path)
    input_img_h, input_img_w = input_img.shape[:2]
    # Add transparency to the input image.
    input_img = np.dstack(
        [input_img, np.ones((input_img_h, input_img_w), dtype='uint8') * 255]
    )
    # The processed images are used in suitable area checks.
    gr_input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    blurred_gr_input_img = cv2.GaussianBlur(gr_input_img, (5, 5), 0)
    edges_input_img = cv2.Canny(
        blurred_gr_input_img, c.CANNY_THRESHOLD_1, c.CANNY_THRESHOLD_2
    )
    # Detecting words.
    unsuitable_blocks = set()
    if args.use_OCR:
        print(f'{input_img_path} text detecting in progress...')
        data = pytesseract.image_to_data(input_img, lang='eng+rus')
        for word_info in data.splitlines()[1:]:
            word_info = word_info.split()
            if len(word_info) == 12:
                x, y, width, height = map(int, word_info[6:10])
                unsuitable_blocks.add((x, y, x + width, y + height))

    suitable_areas = find_suitable_areas(
        unsuitable_blocks, input_img_w, input_img_h, gr_input_img,
        edges_input_img
    )
    if len(suitable_areas) != 0:
        # Watermark partial coverage.
        areas_to_use_amount = round(round(args.used_areas_percentage / 100, 3) * len(suitable_areas))
        # If areas_to_use_amount is 0, make it 1.
        areas_to_use_amount = areas_to_use_amount or 1
        shuffle(suitable_areas)
        suitable_areas = suitable_areas[:areas_to_use_amount]

        # Draw suitable area outlines.
        if c.SHOW_WMARK_AREA_OUTLINES:
            for s_area in suitable_areas:
                cv2.rectangle(
                    input_img, s_area['rect_coords'][:2],
                    s_area['rect_coords'][2:],
                    (255, 0, 0), 1
                )

        # Insert one or more watermarks.
        overlay = np.zeros((input_img_h, input_img_w, 4), dtype='uint8')
        for s_area in suitable_areas:
            overlay[
                s_area['rect_coords'][1]:s_area['rect_coords'][3],
                s_area['rect_coords'][0]:s_area['rect_coords'][2]
            ] = scaled_wmark_imgs[s_area['wmark_scale']]

        # Make black areas where watermark will be inserted.
        (ov_B, ov_G, ov_R, ov_A) = cv2.split(overlay)
        (img_B, img_G, img_R, img_A) = cv2.split(input_img)
        ov_A_wm_absence = ov_A == 0
        input_img = cv2.merge([
            img_B * ov_A_wm_absence, img_G * ov_A_wm_absence,
            img_R * ov_A_wm_absence, img_A
        ])
        # Create image by cutting out watermark areas in input image.
        ov_A_wm_presence = ov_A != 0
        img_B *= ov_A_wm_presence
        img_G *= ov_A_wm_presence
        img_R *= ov_A_wm_presence
        inp_img_cutout = cv2.merge([img_B // 2, img_G // 2, img_R // 2, img_A])
        # cv2.imshow(f'inp_img_cutout for {input_img_path}', inp_img_cutout)

        # Blend the input image with the overlay.
        cv2.addWeighted(overlay, .7, input_img, 1.0, 0, input_img)
        cv2.addWeighted(inp_img_cutout, .7, input_img, 1.0, 0, input_img)
        # cv2.imshow(f'result for {input_img_path}', input_img)
        # cv2.waitKey(0)

        # Save image with inserted watermark.
        inpt_img_name = input_img_path[input_img_path.rfind(os.path.sep) + 1:]
        result_img_path = os.path.join(
            c.PROGRAM_DIR, args.output_dir,
            f'{inpt_img_name.partition(".")[0]}_processed.png'
        )
        cv2.imwrite(result_img_path, input_img)
    else:
        print(f'WARNING: image {input_img_path} has no suitable space to insert watermarks!')


if __name__ == '__main__':
    if args.use_OCR:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path
    # Check paths for existence.
    if not os.path.exists(args.wm_path):
        print('ERROR: Such watermark image file doesn\'n exist!')
        exit(1)
    if not os.path.exists(args.input_dir):
        print(f'ERROR: "{args.input_dir}" directory doesn\'t exist! Specify '
              'another path for input images directory or create it and move '
              'images there.')
        exit(2)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Read watermark image.
    wmark_img = cv2.imread(args.wm_path, cv2.IMREAD_UNCHANGED)
    (wm_h, wm_w) = wmark_img.shape[:2]
    # Opacity bug correction.
    (B, G, R, A) = cv2.split(wmark_img)
    B = cv2.bitwise_and(B, B, mask=A)
    G = cv2.bitwise_and(G, G, mask=A)
    R = cv2.bitwise_and(R, R, mask=A)
    wmark_img = cv2.merge([B, G, R, A])

    # Get scaled watermarks.
    scaled_wmark_imgs = get_scaled_watermarks(wmark_img, args.min_wm_scale)

    # Process all images placed in input images directory.
    input_files = os.listdir(args.input_dir)
    # Filter images among all files existing in the directory.
    input_imgs = list(filter(
        lambda file: file[file.rfind('.') + 1:] in c.ACCEPTABLE_FILE_EXTENSIONS,
        input_files
    ))
    if len(input_imgs) == 0:
        print(f'ERROR: "{args.input_dir}" directory has no images!')
        exit(3)

    for input_img_path in input_imgs:
        insert_watermarks(input_img_path)

    print(f'Elapsed time is {time() - c.START_TIME} seconds.', end='\n\n')
