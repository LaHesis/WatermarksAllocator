import cv2
import pytesseract


def rects_overlapp(r1coords, r2coords):
    if r1coords[0] >= r2coords[2] or r1coords[2] <= r2coords[0] \
            or r1coords[3] <= r2coords[1] or r1coords[1] >= r2coords[3]:
        return False
    return True

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img = cv2.imread('input.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_h, img_w, _ = img.shape

# Detecting words.
data = pytesseract.image_to_data(img, lang='eng+rus')
blocks_with_text = []
for word_info in data.splitlines()[1:]:
    word_info = word_info.split()
    if len(word_info) == 12:
        x, y, width, height = map(int, word_info[6:10])
        blocks_with_text.append((x, y, x + width, y + height))
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
        cv2.putText(
            img, word_info[11], (x, y), cv2.FONT_HERSHEY_COMPLEX,
            .4, (0, 0, 255), 1
        )

mark_width = 150
mark_height = 100
step = 10
min_scale = .2
for s_i in range(10, int(min_scale * 10) + 1, -1):
    cur_scl_blocks_with_text = blocks_with_text
    mark_scale = s_i / 10
    scaled_mark_width = int(mark_width * mark_scale)
    scaled_mark_height = int(mark_height * mark_scale)
    # print(f'mark_scale is {mark_scale}')
    marks_in_img_along_x = img_w // (scaled_mark_width * mark_scale)
    # print(f'marks_in_img_along_x is {marks_in_img_along_x}')
    if marks_in_img_along_x > 0:
        y = 0
        while y < img_h:
            x = 0
            cur_scl_blocks_with_text = list(filter(
                lambda bl: bl[3] >= y,
                cur_scl_blocks_with_text
            ))
            while x < img_w:
                potential_mark_rect = (
                    x, y, x + scaled_mark_width,
                    y + scaled_mark_height
                )
                # Check for overlapping with text.
                for block in cur_scl_blocks_with_text:
                    if rects_overlapp(potential_mark_rect, block):
                        break
                else:
                    cv2.rectangle(
                        img, potential_mark_rect[:2],
                        potential_mark_rect[2:],
                        (255, 0, 0), 1
                    )
                    x += scaled_mark_width - step
                    blocks_with_text.append(potential_mark_rect)
                    cur_scl_blocks_with_text.append(potential_mark_rect)
                x += step
            y += step

cv2.imshow('Result', img)
cv2.imwrite('result.png', img)
cv2.waitKey(0)
