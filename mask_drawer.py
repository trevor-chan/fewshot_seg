import os
import cv2
import numpy as np
import PIL
from PIL import Image
from pathlib import Path


button_pressed = None
state = 'pos'

radius = 10

output_size=(224,224)


def draw_circle(event,x,y,flags,masks):
    
    global button_pressed, radius
    pos_neg,mask = masks
    if event == cv2.EVENT_LBUTTONDOWN:
        button_pressed = 'Left'
    if event == cv2.EVENT_RBUTTONDOWN:
        button_pressed = 'Right'
    if event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        button_pressed = None
    if event == cv2.EVENT_MOUSEMOVE and button_pressed == 'Left':
        if state == 'pos':
            cv2.circle(pos_neg,(x,y),radius,(0,0,255),-1)
        else:
            cv2.circle(pos_neg,(x,y),radius,(255,0,0),-1)
        cv2.circle(mask,(x,y),radius,(0,0,255),-1)
    if event == cv2.EVENT_MOUSEMOVE and button_pressed == 'Right':
        cv2.circle(mask,(x,y),radius,(0,0,0),-1)
        cv2.circle(pos_neg,(x,y),radius,(0,0,0),-1)




def one_image (input_path):
    global radius, state

    background = np.array(cv2.imread(str(input_path)))
    mask = np.zeros(background.shape, np.uint8)
    pos_neg = np.zeros(background.shape, np.uint8)

    cv2.namedWindow('image', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('image',draw_circle,(pos_neg,mask))



    while(1):
        added_image = cv2.addWeighted(background,1.,pos_neg,0.4,0)
        cv2.imshow('image',added_image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        if k == ord(']'):
            radius += 5
        if k == ord('['):
            radius -= 5
            if radius < 5:
                radius = 5
        if k == ord('='):
            state = 'pos'
        if k == ord('-'):
            state = 'neg'

    cv2.destroyAllWindows()


    output_path = Path('./labeler_output/')

    PIL.Image.open(input_path).resize(output_size).convert('RGB').save(output_path / input_path.with_suffix('').with_suffix('.jpg').name)
    PIL.Image.fromarray(pos_neg[:, :, 2]).resize(output_size).convert('RGBA').save (output_path / (input_path.with_suffix('').name + '_posneg.png'))
    PIL.Image.fromarray(mask[:, :, 2]).resize(output_size).convert('RGBA').save (output_path / input_path.with_suffix('').with_suffix('.png').name)


if __name__ == '__main__':
	os.makedirs('labeler_output', exist_ok=True)
	for input_image in [x for x in Path('./labeler_input').iterdir() if x.is_file()]:

	    print (input_image)
	    try:
	        one_image (input_image)
	    except Exception as e:
	        print (f'something went wrong trying to parse image: {input_image}')
	        print (e)
