import time
from pyzbar import pyzbar
import cv2
import mss
import numpy
from rotation import get_rotation
import os

def draw_barcode(decoded, image):
    # n_points = len(decoded.polygon)
    # for i in range(n_points):
    #     image = cv2.line(image, decoded.polygon[i], decoded.polygon[(i+1) % n_points], color=(0, 255, 0), thickness=5)
    # uncomment above and comment below if you want to draw a polygon and not a rectangle
    image = cv2.rectangle(image, (decoded.rect.left, decoded.rect.top), 
                            (decoded.rect.left + decoded.rect.width, decoded.rect.top + decoded.rect.height),
                            color=(0, 255, 0),
                            thickness=5)
    return image

def decode(image):
    # decodes all barcodes from an image
    decoded_objects = pyzbar.decode(image,[2, 5])
    # if len(decoded_objects) == 0:
    #     print('Its clouded :(')
    for obj in decoded_objects:
        # draw the barcode
        # print("detected barcode:", obj)
        image = draw_barcode(obj, image)
        # print barcode type & data
        # print("Type:", obj.type)
        # print("Data:", obj.data)
        # print()
        if obj.data == b'33333':
            print('ROT 1')
        if obj.data == b'66666':
            print('ROT 2')
        if obj.data == b'99999':
            print('ROT 3')

    return image, len(decoded_objects) == 0

if __name__ == "__main__":
    from glob import glob
        
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {"top": 0, "left": 0, "width": 1000, "height": 1080}

        while "Screen capturing":
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))
            # cv2.imshow("img", img)
            # Display the picture
            found = False
            for deg in range(0,90,10):
                
                img_rot = get_rotation(img,deg)
                img_rot, this_found = decode(img_rot)
                # print("FOUND AT DEG {}".format(deg))
                # show the image
                if not this_found:
                    found = True
                    break
                
            if not found:
                print('Its clouded :(')
                # cv2.imshow("OpenCV/Numpy normal", img)

                # print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
            



    # barcodes = glob("barcode*.png")
    # for barcode_file in barcodes:
    #     # load the image to opencv
    #     img = cv2.imread(barcode_file)
    #     # decode detected barcodes & get the image
    #     # that is drawn
    #     img = decode(img)
    #     # show the image
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)