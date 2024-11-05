import streamlit as st
from ultralytics import YOLO
import os
import zipfile
import yaml
import subprocess
import configtrain
import time
import config
import base64
from pathlib import Path
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
# Setting page layout
st.set_page_config(
    page_title="YOLO训练",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

task_type = st.sidebar.selectbox(
    "选择任务",
    ["训练", "检测"]
)
model_pre="/root/yolo/yolo11n.pt"
# Store training process info
if "training_process" not in st.session_state:
    st.session_state.training_process = None

# Function to stop training process
def stop_training():
    if st.session_state.training_process:
        st.session_state.training_process.terminate()
        st.session_state.training_process.wait()  # Ensure the process has completely stopped
        st.success("训练已停止，并回收了资源。")
        st.session_state.training_process = None
    else:
        st.error("没有正在运行的训练进程。")
def get_binary_file_downloader_html(bin_file, download_filename, button_text="Download"):
    with open(bin_file, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()  # Convert to base64
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{download_filename}">{button_text}</a>'
    return href
# Display log in Streamlit
def display_log(logfile):
    st.text("训练日志:")
    if os.path.exists(logfile):
        with open(logfile, "r") as f:
            st.text_area("日志内容", f.read(), height=300)

if task_type == "训练":
    
    # Main page heading
    st.title("YOLO 模型训练")
    
    # Sidebar for dataset template download
    st.sidebar.header("下载数据集模板")
    if st.sidebar.button("下载模板"):
        template_zip_path = '/root/yolo/ultralytics/teample/datasets.zip'  # Update your template file path
        if os.path.exists(template_zip_path):
            st.sidebar.markdown(get_binary_file_downloader_html(template_zip_path, '数据集模板.zip', button_text="下载数据集模板"), unsafe_allow_html=True)
        else:
            st.sidebar.error("模板文件不存在。请检查文件路径。")
        # st.sidebar.markdown(get_binary_file_downloader_html(template_zip_path, '数据集模板'), unsafe_allow_html=True)
    
    # Sidebar for training configuration
    st.sidebar.header("训练配置")
    model_type = st.sidebar.selectbox(
        "选择模型",
        ["yolov8n", "yolov5", "yolov6", "yolov10", "yolov11", "Rtdetr"]
    )
    
    # Model paths based on selected model type
    model_yaml_map = {
        'yolov8n': "v8/yolov8.yaml",
        'yolov5': "v5/yolov5.yaml",
        'yolov6': "v6/yolov6.yaml",
        'yolov10': "v10/yolov10n.yaml",
        'yolov11': "11/yolo11.yaml",
        'Rtdetr': "rt-detr/rtdetr-resnet50.yaml"
    }
    path = model_yaml_map.get(model_type)
    # Training options
    epochs = st.sidebar.number_input("训练轮数", min_value=1, max_value=1000, value=150)
    batch_size = st.sidebar.number_input("批量大小", min_value=1, max_value=64, value=16)
    img_size = st.sidebar.number_input("图片大小", min_value=32, max_value=1024, value=640)
    workers = st.sidebar.number_input("工作进程数", min_value=1, max_value=8, value=2)
    
    # Category options
    categories = st.sidebar.text_input("输入类别名称（用逗号分隔）", value="tomato")
    categories_list = [cat.strip() for cat in categories.split(',')]
    
    # Model and data configuration
    model_yaml = "/root/yolo/ultralytics/ultralytics/cfg/models/" + path

    data_yaml = "/root/yolo/ultralytics/ultralytics/cfg/datasets/coco.yaml"
    
    # Upload dataset
    st.sidebar.header("上传数据集")
    uploaded_zip_file = st.sidebar.file_uploader("选择数据集压缩文件", type=['zip'], accept_multiple_files=False)
    
    # Unzip dataset to a specific folder
    def unzip_dataset(zip_file, extract_path):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Modify the data_yaml file with the new categories
    def modify_yaml_file(file_path, categories):
        names_dict = {i: cat for i, cat in enumerate(categories)}
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        data['names'] = names_dict
        with open(file_path, 'w') as file:
            yaml.dump(data, file)

    def modify_datayaml_file(file_path, count):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        data['nc'] = count
        with open(file_path, 'w') as file:
            yaml.dump(data, file)

    # Prepare the dataset paths for training
    def prepare_dataset(pic_path, save_txtfile):
        file_exists = os.path.isfile(save_txtfile)

        # Create an empty file if it does not exist
        if not file_exists:
            with open(save_txtfile, 'w') as file:
                pass  # Create an empty file
        with open(save_txtfile, 'w') as save_txtfile:
            for root, dirs, files in os.walk(pic_path):
                for file in files:
                    save_txtfile.write(pic_path + '/' + file + '\n')

    # Function to start training using YOLO as a subprocess
    def start_training_subprocess(data_yaml, epochs, img_size, batch_size, workers, model_yaml):
        log_file = "/root/yolo/train_log.txt"
        if 'rtdetr' in model_yaml:
             command = f"python3 -c 'from ultralytics import RTDETR; model = RTDETR(\"{model_yaml}\"); model.train(data=\"{data_yaml}\", epochs={epochs}, imgsz={img_size}, batch={batch_size}, workers={workers})' > {log_file} 2>&1"
        else:
        # command = f"python3 -m ultralytics train --data {data_yaml} --epochs {epochs} --imgsz {img_size} --batch {batch_size} --workers {workers} --model {model_yaml} > {log_file} 2>&1"
            command = f"python3 -c 'from ultralytics import YOLO; model = YOLO(\"{model_yaml}\").load(\"{model_pre}\"); model.train(data=\"{data_yaml}\", epochs={epochs}, imgsz={img_size}, batch={batch_size}, workers={workers})' > {log_file} 2>&1"
    
    # 启动训练进程
        st.session_state.training_process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)  # Start as subprocess

    # Monitor the training process
        progress_placeholder = st.empty()  # Placeholder for displaying training status
        while True:
            time.sleep(5)  # Wait before checking the log file
            return_code = st.session_state.training_process.poll()  # Check if the process has completed
            
            # Update the progress status
            if return_code is None:
                progress_placeholder.text("训练进行中...请稍候。")
            else:
                progress_placeholder.text("训练完成！")
                
                break  # Exit the loop if training is complete


        return log_file

    # Start training button
    if st.sidebar.button("开始训练"):
        if uploaded_zip_file is not None:
            dataset_folder = "/root/yolo/ultralytics/dataset"
            os.makedirs(dataset_folder, exist_ok=True)
            zip_path = os.path.join(dataset_folder, uploaded_zip_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip_file.getbuffer())
            unzip_dataset(zip_path, dataset_folder)
            os.remove(zip_path)
            modify_yaml_file(data_yaml, categories_list)
            st.success("处理数据开始，请稍候...")
            prepare_dataset(r"/root/yolo/ultralytics/dataset/datasets/train/images", r"/root/yolo/ultralytics/dataset/datasets/train.txt")
            prepare_dataset(r"/root/yolo/ultralytics/dataset/datasets/valid/images", r"/root/yolo/ultralytics/dataset/datasets/valid.txt")
            st.success("处理完成，请稍候...")
            modify_datayaml_file(model_yaml, len(categories_list))
            
            # Start training and display log
            st.success("训练开始，请稍候...")
            log_file = start_training_subprocess(data_yaml, epochs, img_size, batch_size, workers, model_yaml)
            st.session_state.log_file = log_file
            st.success("训练完成，请稍候...")
            # display_log(log_file)
        else:
            st.error("请上传数据集")
    
    # Add button to stop training process
    if st.sidebar.button("停止训练"):
        stop_training()

    # Display the log file continuously
    if "log_file" in st.session_state:
        display_log(st.session_state.log_file)

elif task_type == "检测":
    # main page heading
    st.title("目标检测系统")
    
    # sidebar
    st.sidebar.header("模型选择")
    
    # model options
    task_type = st.sidebar.selectbox(
        "选择任务",
        ["检测"]
    )
    
    model_type = None
    if task_type == "检测":
        model_type = st.sidebar.selectbox(
            "选择模型",
            config.DETECTION_MODEL_LIST
        )
    else:
        st.error("Currently only 'Detection' function is implemented")
    
    confidence = float(st.sidebar.slider(
        "选择模型置信度", 30, 100, 50)) / 100
    
    model_path = ""
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")
    
    # load pretrained DL model
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {model_path}")
    
    # image/video options
    st.sidebar.header("数据源配置")
    source_selectbox = st.sidebar.selectbox(
        "选择数据源",
        config.SOURCES_LIST
    )
    
    source_img = None
    if source_selectbox == config.SOURCES_LIST[0]: # Image
        infer_uploaded_image(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[1]: # Video
        infer_uploaded_video(confidence, model)
    elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
        infer_uploaded_webcam(confidence, model)
    else:
        st.error("Currently only 'Image' and 'Video' source are implemented")