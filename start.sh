echo "setting up env..."
pip install requests pyyaml -t /usr/local/lib/python3.5/dist-packages && clear && source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5 &>> log.txt
clear

echo "running mmqt server..."
cd webservice/server/node-server
node ./server.js &>> log.txt
clear 

echo "running UI..."
cd ../../ui
npm run dev &>> log.txt
clear

echo "running ffmpeg server..."
cd ../..
sudo ffserver -f ./ffmpeg/server.conf &>> log.txt
clear

echo "running people counter app..."
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
clear
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/optimised/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.1 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
