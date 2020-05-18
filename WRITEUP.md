# Project Write-Up

## Explaining Custom Layers
For my requirements I did need any custom layers to be created for this people couter app. I have used ssd mobile net V2 coco tensorflow model and it worked very well.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were checking difference betweeen model accuray, size and inference time. I have tried this by running the model without open vino and with open vino toolkit with same image. And got following noticable differences-

- The difference between model accuracy pre- and post-conversion was 0.77801156 and 0.8787151 respectively.
- The size of the model pre- and post-conversion was 67 MB and 65 MB respectively.
- The inference time of the model pre- and post-conversion was 3547 ms and 72 ms respectively.

## Assess Model Use Cases

### To alert police or govt. for people in max geathering in this COVID-19 situation
This people counter app can be used in cities where government is trying to restrict people geathering in huge amout. So it can directly send an sms or picture of people and count and also the location. So the Police don't have to roam everywhere to control the rush. Then can only take action when there is an alert. It will save life of police and public also.

### To Control and Alert Management in Lift
This a very basic use case that we can have. In lifts generally we have fix capacity for no of people. So This system can detect the no of people and alert or stop the lift to move and Notify the people to reduce the no of people. 

### As an tracker for capacity for some event of conference
This can be also very useful when there an event or conference and we have fixed no of seats. So we can alert security officers if more then expected people are coming.

### At traffic lights to figure out max busy road
This I though because there are some roads or signals where we have lot of traffic whole day or specially office hours. So We can analysis these places by no of people and no of time duratoin they spend to figure out new flyovers or new roads that can built there as alternative and to divide traffic.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows-

- Lighting is required to work model effectively. In low light or no light It may not give accurate output. So this can effect the performance of model but if can be overcome by keeping lights on and also we can use camera's which can capture videos or images in dark also.

- Model Accuracy and speed are to factor that combiningly effect the end user. One must choose either one based on the requirements.

- Focal length or image size is also important factor for model accuracy. Focal length is also related with light. So better the focal length better will be result of image and model output. And Smaller image will decrease the model accuracy so image size should be good enough to detect people.

## Model Research

In investigating potential people counter models, I tried each of the following three model:

- Model: SSD Mobile Net V2 COCO
  - I have found the model at [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
  
  - I have created a new directory for models by following command-
  ```
  mkdir models
  ```
  
  - Downloaded the model from tensorflow detection model zoo-
  ```
  cd models
  wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```
  
  - extracted model files from tar file using below command-
  ```
  tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  ```
  
  - I converted the model to an Intermediate Representation with the following arguments-
  ```
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_mobilenet_v2_coco_2018_03_29/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json -o optimized/ssd_mobilenet_v2_coco_2018_03_29
  ```
  
  - Source the env by following command-
  ```
  source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
  ```
  
  - To run the inferance on model I used below command-
  ```
  python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/optimized/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.1 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
  ```