import cv2
import pytesseract
import os
# import numpy as np
from progress.spinner import Spinner
from time import time

start_t = time()

MARK_WIDTH = 150
MARK_HEIGHT = 100
STEP = 20
MIN_SCALE = .3
CANNY_THRESHOLD_1 = 120
CANNY_THRESHOLD_2 = 120
# If an image area contains pixels with a lower value, no watermark
# would be placed there. Should be between 1 and 255.
NO_WMARK_DARK_MAX_VAL = 80
NO_WMARK_MIN_CANNY_PIXELS_AMOUNT = 10
NO_WMARK_MIN_DARK_PIXELS_AMOUNT = 20

def rects_overlapp(r1coords, r2coords):
    if r1coords[0] >= r2coords[2] or r1coords[2] <= r2coords[0] \
            or r1coords[3] <= r2coords[1] or r1coords[1] >= r2coords[3]:
        return False
    return True

print('Text detecting...')
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('input.jpeg')
gr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blurred_gr_img = cv2.GaussianBlur(gr_img, (5, 5), 0)
canny_img = cv2.Canny(blurred_gr_img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
# cv2.imshow('blurred', blurred_gr_img)
# cv2.imshow('canny', canny_img)
cv2.waitKey(0)

img_h, img_w, _ = img.shape

# TODO: program arguments.
# TODO: Tesseract optional use.
# TODO: batch images processing.

# Detecting words.
data = pytesseract.image_to_data(img, lang='eng+rus')
blocks_with_text = set()
for word_info in data.splitlines()[1:]:
    word_info = word_info.split()
    if len(word_info) == 12:
        x, y, width, height = map(int, word_info[6:10])
        blocks_with_text.add((x, y, x + width, y + height))
        # cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
        # cv2.putText(
        #     img, word_info[11], (x, y), cv2.FONT_HERSHEY_COMPLEX,
        #     .4, (0, 0, 255), 1
        # )

for s_i in range(10, int(MIN_SCALE * 10) - 1, -1):
    cur_scl_blocks_with_text = list(blocks_with_text)
    mark_scale = s_i / 10
    spinner = Spinner(
        f'Available space estimating for watermark scale {mark_scale}...',
    )
    scaled_mark_width = int(MARK_WIDTH * mark_scale)
    scaled_mark_height = int(MARK_HEIGHT * mark_scale)
    # print(f'mark_scale is {mark_scale}')
    marks_in_img_along_x = img_w // (scaled_mark_width * mark_scale)
    # print(f'marks_in_img_along_x is {marks_in_img_along_x}')
    if marks_in_img_along_x > 0:
        y = 0
        while y + scaled_mark_height < img_h:
            x = 0
            cur_scl_blocks_with_text = list(filter(
                lambda bl: bl[3] >= y,
                cur_scl_blocks_with_text
            ))
            while x + scaled_mark_width < img_w:
                potential_mark_rect = (
                    x, y, x + scaled_mark_width,
                    y + scaled_mark_height
                )
                overlapped_blocks = list(filter(
                    lambda bl: rects_overlapp(bl, potential_mark_rect),
                    cur_scl_blocks_with_text
                ))
                if len(overlapped_blocks) == 0:
                    gr_roi = gr_img[y:y + scaled_mark_height, x:x + scaled_mark_width].flatten()
                    canny_roi = canny_img[y:y + scaled_mark_height, x:x + scaled_mark_width].flatten()
                    no_wmark_in_dark = (gr_roi < NO_WMARK_DARK_MAX_VAL).sum() > NO_WMARK_MIN_DARK_PIXELS_AMOUNT
                    no_wmark_in_canny = (canny_roi > 0).sum() > NO_WMARK_MIN_CANNY_PIXELS_AMOUNT
                    # if not area_with_canny and not area_with_dark:
                    if not no_wmark_in_dark and not no_wmark_in_canny:
                        cv2.rectangle(
                            img, potential_mark_rect[:2],
                            potential_mark_rect[2:],
                            (255, 0, 0), 1
                        )
                        x += scaled_mark_width - STEP
                        blocks_with_text.add(potential_mark_rect)
                        cur_scl_blocks_with_text.append(potential_mark_rect)
                x += STEP
            y += STEP
            spinner.next()
            # TODO: filter out blocks below y.
    spinner.finish()

cv2.imshow('Result', img)
cv2.imwrite('result.png', img)
print(f'Elapsed time is {time() - start_t} seconds.')
cv2.waitKey(0)
