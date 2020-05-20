# Project Write-Up

## Explaining Custom Layers
The process behind converting custom layer for TensorFlow, we have two options-
- Register the custom layer as model extensions to model optimizer so that model optimizer can convert a valid intermediate representation.

- If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models. More details we can find on open vino doc for subgraph replacement.

We need custom layers when existing layers are not enough to provide the desired output. Generally When we need an extra feature to be added in the model for prediction which is already not there in the existing trained model. We can combine two or more existing layers to form a new layer. The custom layer have it's own weight and transformation logic to be performed on the tensor. When we covert existing model from a different framework like Tensorflow, ONNX, Caffe, etc the model optimizer check all the of layers of the model in it's known layer list depending on the framework. Because different framework supports different layers. So If we have added any custom layer we need to add extensions or we need to register that layer so that model optimizer can covert that unsupported layer and produce intermediate representation from the model. In the real world, we generally need many custom layers like When we want to build a recognition model in that If we want to specify a unique feature that is not there in the existing model to improve the accuracy and confidence of the model.

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were checking the difference between model accuracy, size, and inference time. I have tried this by running the model without open vino and with an open vino toolkit with the same image. And got following noticeable differences-

- The difference between model accuracy pre- and post-conversion was 0.77801156 and 0.8787151 respectively.
- The size of the model pre- and post-conversion was 67 MB and 65 MB respectively.
- The inference time of the model pre- and post-conversion was 3547 ms and 72 ms respectively.

### Edge Vs Cloud
Edge computing we require when there are some areas where there is a low network or we need the data, alert, or message from IoT devices at some specific condition. In such a case we can deploy the model at Edge and it will send the required data only when the conditions meet. This will save the network calls and cost as well because if we use cloud computing in these situations then for every processing we need to make network calls from IoT devices to cloud server and also the cloud machine will keep running all the time so service provider will charge for that which will be too costly because there will be huge data generated from IoT device but in case of Edge Computing we will process the data on edge only and only send the processed data as output we need to store in the cloud and for that obviously the cost will be very less. So the conclusion is that Edge computing requires very less network and less processing resources which will save the cost.

## Assess Model Use Cases

### To alert police or govt. for people in max gathering in this COVID-19 situation
This people counter app can be used in cities where the government is trying to restrict people from gathering in huge amounts. So it can directly send an SMS or picture of people and count and also the location. So the Police don't have to roam everywhere to control the rush. Then can only take action when there is an alert. It will save the life of the police and the public also.

### To Control and Alert Management in Lift
This a very basic use case that we can have. In lifts generally, we have fix capacity for no of people. So This system can detect the no of people and alert or stop the lift to move and Notify the people to reduce the no of people. 

### As a tracker for capacity for some event of the conference
This can be also very useful when there an event or conference and we have fixed no. of seats. So we can alert security officers if more then expected people are coming.

### At traffic lights to figure out a max busy road
This I though because there are some roads or signals where we have a lot of traffic the whole day or especially office hours. So We can analyze these places by no. of people and no. of time duration they spend to figure out new flyovers or new roads that can build there as alternatives and divide traffic.

## Assess Effects on End-User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows-

- Lighting is required to work model effectively. In low light or no light, It may not give accurate output. So this can affect the performance of the model but it can be overcome by keeping lights on and also we can use cameras that can capture videos or images in dark also.

- Model Accuracy and speed are the factor that combining affect the end-user. One must choose either one based on the requirements.

- Focal length or image size is also important to factor for model accuracy. The focal length is also related to light. So better the focal length better will be the result of image and model output. And Smaller image will decrease the model accuracy so image size should be good enough to detect people.

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