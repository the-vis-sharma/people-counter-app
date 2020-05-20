"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import numpy as np
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network


# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

supported_img_ext = [".jpg", ".jpeg", ".png", ".bmp"]
supported_vd_ext = [".mp4", ".avi"]
def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def draw_masks(frame, result, width, height, prob_threshold):
    people = 0
    probabilities = result[0, 0, :, 2]
    out_frame = frame
    x1 = None
    for index, probability in enumerate(probabilities):
        if probability > prob_threshold:
            people += 1
            x1, y1, x2, y2 = result[0, 0, index, 3:]
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            out_frame = cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
            
    out_frame = cv2.resize(out_frame, (768, 432))
    return out_frame, people, x1


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(model=args.model, cpu_extension=args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    # Handle the input stream
    inp = None
    single_image_model = False
    if args.input == 'CAM':
        inp = 0
    elif os.path.isfile(args.input):
        ext = os.path.splitext(args.input)[1]
        if ext in supported_img_ext:
            single_image_model = True
            inp = args.input
        elif ext in supported_vd_ext:
            inp = args.input
        else:
            log.error("This file format not support yet.")
            exit(1)
    else:
        log.error("This file not found in your system. Please make sure the file exists.")
        exit(1)
        
        
    cap = cv2.VideoCapture(inp)
    cap.open(inp)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Loop until stream is over
    total_duration = 0
    noPersonDetected = 0
    total_people = 0
    start_time = None
    last_frame_with_person = None
    last_count = None
    last_x1 = 0
    timer = 0
    while cap.isOpened():
        # Read from the video capture
        flag, frame = cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        # Pre-process the image as needed
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose(2, 0, 1)
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Start asynchronous inference for specified request
        infer_network.exec_net(p_frame)

        # Wait for the result
        if infer_network.wait() == 0:
            # Get the results of the inference request
            result = infer_network.get_output()
            
            # Extract any desired stats from the results        
            out_frame, people, x1 = draw_masks(frame, result, width, height, args.prob_threshold)
            
            if start_time is None and people > 0:
                start_time = timer
            
            # Calculate and send relevant information on current_count, total_count and duration to the MQTT server Topic "person": keys of "count" and "total" Topic "person/duration": key of "duration"
            if people > 0:
                noPersonDetected = 0
                last_frame_with_person = out_frame
                last_count = people
                last_x1 = x1
            else:
                noPersonDetected += 1
                    
                if start_time is not None and noPersonDetected == 1:
                    total_duration = timer - start_time
                
                if last_frame_with_person is not None and last_x1 <= 0.75:
                    out_frame = last_frame_with_person
                    people = last_count
                    
            if start_time is not None and noPersonDetected >= 15:
                total_people += last_count
                client.publish("person/duration", json.dumps({"duration": total_duration}))
                start_time = None
                people = 0
            
            client.publish("person", json.dumps({"count": people, "total": total_people}))

        # Send the frame to the FFMPEG server
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        # Write an output image if `single_image_mode`
        if single_image_model:
            cv2.imwrite("output" + ext, out_frame)
        
        if key_pressed == 27:
            break
            
        timer += 1
    # release the cam and destory any camera window
    cap.release()
    cv2.destroyAllWindows()
    # disconnect MQTT server.
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
