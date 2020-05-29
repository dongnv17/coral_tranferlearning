# coral_tranferlearning
Tranfer learning an object detection model to coral google (Mobilenet v2 ssd)

## 1. Tải weight của model ssd_mobilenet_v2

wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz

## 2. Giải nén vào thư mục learn_pet/ckpt
Thay đổi nội dung file pipeline.config như sau:
```
- num_classes: 2: Số class train
- type: "ssd_mobilenet_v2": Tên mạng
- fine_tune_checkpoint: "/mnt/DATA/tensorflow/models/research/learn_pet/ckpt/model.ckpt": Đường dẫn đến checkpoint ssd_mobilenet_v2 ở trên
- label_map_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_label_map.pbtxt": Đường dẫn đến label map
- <train_input_reader> input_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_train.record": Đường dẫn output của TFRecord tập train data
- <eval_input_reader>input_path: "/mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_val.record": : Đường dẫn output của TFRecord tập validatation data
```

## 3. Cài đặt các gói cho python (tensorflow, Pillow, lxml, protobuf, ...)

* absl-py==0.6.0
* astor==0.7.1
* backports-abc==0.5
* backports.functools-lru-cache==1.5
* backports.shutil-get-terminal-size==1.0.0
* backports.weakref==1.0.post1
* bleach==3.0.2
* configparser==3.5.0
* contextlib2==0.6.0.post1
* cycler==0.10.0
* Cython==0.29.19
* decorator==4.3.0
* defusedxml==0.5.0
* entrypoints==0.2.3
* enum34==1.1.6
* funcsigs==1.0.2
* functools32==3.2.3.post2
* futures==3.2.0
* gast==0.2.0
* grpcio==1.16.0
* h5py==2.8.0
* ipaddress==1.0.22
* ipykernel==4.10.0
* ipython==5.8.0
* ipython-genutils==0.2.0
* ipywidgets==7.4.2
* Jinja2==2.10
* jsonschema==2.6.0
* jupyter==1.0.0
* jupyter-client==5.2.3
* jupyter-console==5.2.0
* jupyter-core==4.4.0
* Keras-Applications==1.0.6
* Keras-Preprocessing==1.0.5
* kiwisolver==1.0.1
* lxml==4.5.1
* Markdown==3.0.1
* MarkupSafe==1.0
* matplotlib==2.2.3
* mistune==0.8.4
* mock==2.0.0
* nbconvert==5.4.0
* bformat==4.4.0
* notebook==5.7.0
* numpy==1.15.3
* pandas==0.23.4
* pandocfilters==1.4.2
* pathlib2==2.3.2
* pbr==5.1.0
* pexpect==4.6.0
* pickleshare==0.7.5
* Pillow==5.3.0
* prometheus-client==0.4.2
* prompt-toolkit==1.0.15
* protobuf==3.6.1
* ptyprocess==0.6.0
* Pygments==2.2.0
* pyparsing==2.2.2
* python-dateutil==2.7.4
* pytz==2018.6
* pyzmq==17.1.2
* qtconsole==4.4.2
* scandir==1.9.0
* scikit-learn==0.20.0
* scipy==1.1.0
* Send2Trash==1.5.0
* simplegeneric==0.8.1
* singledispatch==3.4.0.3
* six==1.11.0
* sklearn==0.0
* subprocess32==3.5.3
* tensorboard==1.12.0
* tensorflow==1.12.0rc2
* termcolor==1.1.0
* terminado==0.8.1
* testpath==0.4.2
* tornado==5.1.1
* traitlets==4.3.2
* wcwidth==0.1.7
* webencodings==0.5.1
* Werkzeug==0.14.1
* widgetsnbextension==3.4.2


## 4. Chuẩn bị dữ liệu để train
- File label map: /mnt/DATA/tensorflow/models/research/learn_pet/pet/hand_label_map.pbtxt: Ví dụ ở đây có 2 classes - ID phải bắt đầu từ 1
- Folder: images: Chứa các ảnh để train
- Folder: annotations chứa: 
	- File: trainval.txt: Danh sách tên tất cả các file ảnh train và validation
	- Folder xmls: Chứa các annotations của các ảnh đã label
##5. Bắt đầu train (fine-turning model) và đợi
Chạy lệnh: 
```
./retrain_detection_model.sh --num_training_steps ${NUM_TRAINING_STEPS} --num_eval_steps ${NUM_EVAL_STEPS}
```

NUM_TRAINING_STEPS: Số step training (ví dụ 500)

NUM_EVAL_STEPS: Số step validation (Ví dụ 100)
## 6. Convert checkpoint sau khi train sang TFLite và quantize 8bit cho edge Tpu
Khi đang train sẽ bắt đầu sinh ra các checkpoint ở thư mục /mnt/DATA/tensorflow/models/research/learn_pet/train
- Chạy lệnh: 
```
./convert_checkpoint_to_edgetpu_tflite.sh --checkpoint_num 500 (Ví dụ với checkpoint 500)
```

Sau khi chạy lệnh sẽ sinh ra model TFLite ở thư mục 
```
/mnt/DATA/tensorflow/models/research/learn_pet/models/output_tflite_graph_edgetpu.tflite
```

## 7. Compile model ở trên để chạy được trên Edge Tpu
- Cài Edge TPU compiler:
```
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt update

sudo apt-get install edgetpu-compiler
```
- Compile model:
```
cd /mnt/DATA/tensorflow/models/research/learn_pet/models/

edgetpu_compiler output_tflite_graph.tflite
```
Sẽ sinh ra model: output_tflite_graph_edgetpu.tflite để chạy được trên Edge TPU

## Chú ý: Ở đây thư mục chưa project là: /mnt/DATA/tensorflow/models/research và các lệnh được chạy tại thư mục này

