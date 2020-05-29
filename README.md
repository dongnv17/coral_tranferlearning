# coral_tranferlearning
Tranfer learning an object detection model to coral google (Mobilenet v2 ssd)
1. Tải weight của model ssd_mobilenet_v2
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

2. Giải nén vào thư mục learn_pet/ckpt
Thay đổi nội dung file pipeline.config như sau:
- num_classes: 2: Số class train
- type: "ssd_mobilenet_v2": Tên mạng
- fine_tune_checkpoint: "/mnt/DATA/tensorflow/models/research/learn_pet/ckpt/model.ckpt": Đường dẫn đến checkpoint ssd_mobilenet_v2 ở trên
- label_map_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_label_map.pbtxt": Đường dẫn đến label map
- <train_input_reader> input_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_train.record": Đường dẫn output của TFRecord tập train data
- <eval_input_reader>input_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_val.record": : Đường dẫn output của TFRecord tập validatation data

3. Cài đặt các gói cho python (tensorflow, Pillow, lxml, protobuf, ...)

absl-py             0.9.0
astor               0.8.1
contextlib2         0.6.0.post1
cycler              0.10.0
Cython              0.29.19
gast                0.3.3
grpcio              1.29.0
h5py                2.10.0
importlib-metadata  1.6.0
Keras-Applications  1.0.8
Keras-Preprocessing 1.1.2
kiwisolver          1.2.0
lxml                4.5.1
Markdown            3.2.2
matplotlib          2.2.3
numpy               1.18.4
Pillow              5.3.0
pip                 20.1
protobuf            3.12.1
pycocotools         2.0.0
pyparsing           2.4.7
python-dateutil     2.8.1
pytz                2020.1
setuptools          46.1.3
six                 1.15.0
tensorboard         1.12.2
tensorflow          1.12.0
termcolor           1.1.0
Werkzeug            1.0.1
wheel               0.34.2
zipp                3.1.0

4. Chuẩn bị dữ liệu để train
- File label map: /mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_label_map.pbtxt: Ví dụ ở đây có 2 classes - ID phải bắt đầu từ 1
- Folder: images: Chứa các ảnh để train
- Folder: annotations chứa: 
	- File: trainval.txt: Danh sách tên tất cả các file ảnh train và validation
	- Folder xmls: Chứa các annotations của các ảnh đã label
5. Bắt đầu train (fine-turning model) và đợi
Chạy lệnh: ./retrain_detection_model.sh --num_training_steps ${NUM_TRAINING_STEPS} --num_eval_steps ${NUM_EVAL_STEPS}
NUM_TRAINING_STEPS: Số step training (ví dụ 500)
NUM_EVAL_STEPS: Số step validation (Ví dụ 100)
6. Convert checkpoint sau khi train sang TFLite và quantize 8bit cho edge Tpu
Khi đang train sẽ bắt đầu sinh ra các checkpoint ở thư mục /mnt/DATA/tensorflow/models/research/learn_pet/train
- Chạy lệnh: ./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 500 (Ví dụ với checkpoint 500)
Sau khi chạy lệnh sẽ sinh ra model TFLite ở thư mục /mnt/DATA/tensorflow/models/research/learn_pet/models/output_tflite_graph_edgetpu.tflite
7. Compile model ở trên để chạy được trên Edge Tpu
- Cài Edge TPU compiler:
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt update
sudo apt-get install edgetpu-compiler
- Compile model:
cd /mnt/DATA/tensorflow/models/research/learn_pet/models/
edgetpu_compiler output_tflite_graph.tflite
Sẽ sinh ra model: output_tflite_graph_edgetpu.tflite để chạy được trên Edge TPU

Chú ý: Ở đây thư mục chưa project là: /mnt/DATA/tensorflow/models/research và các lệnh được chạy tại thư mục này

