from random import random
from aiogram import Dispatcher, types
from aiogram.dispatcher.filters import Text
import os
import cv2
import time
from datetime import datetime, timedelta
from pixellib.torchbackend.instance import instanceSegmentation
import argparse
import glob
import numpy as np
import os
import torch
import time


#video_name = "video.mp4"
#writer = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), video_speed, (width,height))

def command_person():
    cap = cv2.VideoCapture()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time.sleep(1)
    ret,frame = cap.read()
    name = datetime.now() + timedelta(hours=3)
    #cv2.rectangle(frame, (50, 0), (1000, 720), (0, 255, 0), 2) - рисует прямоугольник
    cv2.imwrite(f"files/{name}.png", frame)
    #writer.write(frame)
    #cv2.imshow('frame', frame) # показываем кадр
    cap.release()
    #writer.release()
    cv2.destroyAllWindows() # закрываем все окна

    time.sleep(0.5)
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")
    #ins.segmentImage(f"files/{name}.png", show_bboxes=True, extract_segmented_objects=True, save_extracted_objects=True, output_image_name="output_image.jpg")
    target_classes = ins.select_target_classes(person = True)
    result, output = ins.segmentImage(f"files/{name}.png", show_bboxes=True, segment_target_classes = target_classes, output_image_name=f"output/output_{name}.jpg")#, extract_segmented_objects=True, segment_target_classes = target_classes, save_extracted_objects=True)
    os.remove(f"files/{name}.png")
    #print(result)
    cap = cv2.VideoCapture()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time.sleep(1)
    ret,frame = cap.read()
    name = datetime.now() + timedelta(hours=3)
    #cv2.rectangle(frame, (50, 0), (1000, 720), (0, 255, 0), 2) - рисует прямоугольник
    cv2.imwrite(f"files/{name}.png", frame)
    #writer.write(frame)
    #cv2.imshow('frame', frame) # показываем кадр
    cap.release()
    #writer.release()
    cv2.destroyAllWindows() # закрываем все окна

    time.sleep(0.5)
    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl")
    #ins.segmentImage(f"files/{name}.png", show_bboxes=True, extract_segmented_objects=True, save_extracted_objects=True, output_image_name="output_image.jpg")
    target_classes = ins.select_target_classes(person = True)
    result, output = ins.segmentImage(f"files/{name}.png", show_bboxes=True, segment_target_classes = target_classes, output_image_name=f"output/output_{name}.jpg")#, extract_segmented_objects=True, segment_target_classes = target_classes, save_extracted_objects=True)
    os.remove(f"files/{name}.png")
    #print(result)
    

    amount = result["object_counts"]["person"]

if __name__ == '__main__':
    
    while True:
        command_person()
        time.sleep(600)