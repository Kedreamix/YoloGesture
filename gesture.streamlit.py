"""Create an Object Detection Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
from yolo import YOLO
import os
import urllib
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
# 设置网页的icon
st.set_page_config(page_title='Gesture Detector', page_icon='✌',
                   layout='centered', initial_sidebar_state='expanded')

RTC_CONFIGURATION = RTCConfiguration(
    {
      "RTCIceServer": [{
        "urls": ["stun:stun.l.google.com:19302"],
        "username": "pikachu",
        "credential": "1234",
      }]
    }
)
def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(open("instructions.md",encoding='utf-8').read())

    
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(open("gesture.streamlit.py",encoding='utf-8').read())
    elif app_mode == "Run the app":
        # Download external dependencies.
        for filename in EXTERNAL_DEPENDENCIES.keys():
            download_file(filename)

        readme_text.empty()
        run_the_app()

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov4_tiny.pth": {
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_tiny.pth",
        "size": 23631189 
    },
    "yolov4_SE.pth": {
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_SE.pth",
        "size": 23806027
    },
    "yolov4_CBAM.pth":{
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_CBAM.pth",
        "size": 23981478
    },
    "yolov4_ECA.pth":{
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_ECA.pth",
        "size": 23632688
    },
    "yolov4_weights_ep150_608.pth":{
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_weights_ep150_608.pth",
        "size": 256423031
    },
    "yolov4_weights_ep150_416.pth":{
        "url": "https://github.com/Kedreamix/YoloGesture/releases/download/v1.0/yolov4_weights_ep150_416.pth",
        "size": 256423031
    },
}


# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return
    # print(os.path.getsize(file_path))
    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))
    except Exception as e:
        print(e)
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():    
    class Config():
        def __init__(self, weights = 'yolov4_tiny.pth', tiny = True, phi = 0, shape = 416,nms_iou = 0.3, confidence = 0.5):
            self.weights = weights
            self.tiny = tiny
            self.phi = phi
            self.cuda = False
            self.shape = shape
            self.confidence = confidence
            self.nms_iou = nms_iou
    # set title of app
    st.markdown('<h1 align="center">✌ Gesture Detection</h1>',
                unsafe_allow_html=True)
    st.sidebar.markdown("# Gesture Detection on?")
    activities = ["Example","Image", "Camera", "FPS", "Heatmap","Real Time", "Video"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    phi = st.sidebar.selectbox("yolov4-tiny 使用的自注意力模式:",('0tiny','1SE','2CABM','3ECA'))
    print("")

    tiny = st.sidebar.checkbox('是否使用 yolov4 tiny 模型')
    if not tiny:
        shape = st.sidebar.selectbox("Choose shape to Input:", [416,608])
    conf,nms = object_detector_ui()
    @st.cache
    def get_yolo(tiny,phi,conf,nms,shape=416):
        weights = 'yolov4_tiny.pth'
        if tiny:
            if phi == '0tiny':
                weights = 'yolov4_tiny.pth'
            elif phi == '1SE':
                weights = 'yolov4_SE.pth'
            elif phi == '2CABM':
                weights = 'yolov4_CBAM.pth'
            elif phi == '3ECA':
                weights = 'yolov4_ECA.pth'
        else:
            if shape == 608:
                weights = 'yolov4_weights_ep150_608.pth'
            elif shape == 416:
                weights = 'yolov4_weights_ep150_416.pth'
        opt = Config(weights = weights, tiny = tiny , phi = int(phi[0]), shape = shape,nms_iou = nms, confidence = conf)
        yolo = YOLO(opt)
        return yolo
    
    if tiny:
        yolo = get_yolo(tiny, phi, conf, nms)
        st.write("YOLOV4 tiny 模型加载完毕")
    else:
        yolo = get_yolo(tiny, phi, conf, nms, shape)
        st.write("YOLOV4 模型加载完毕")
    
    if choice == 'Image':
        detect_image(yolo)
    elif choice =='Camera':
        detect_camera(yolo)
    elif choice == 'FPS':
        detect_fps(yolo)
    elif choice == "Heatmap":
        detect_heatmap(yolo)
    elif choice == "Example":
        detect_example(yolo)
    elif choice == "Real Time":
        detect_realtime(yolo)
    elif choice == "Video":
        detect_video(yolo)
        


# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

def predict(image,yolo):
    """Return predictions.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: none
    """
    crop            = False
    count           = False
    try:
        # image = Image.open(image)
        r_image = yolo.detect_image(image, crop = crop, count=count)
        transform = transforms.Compose([transforms.ToTensor()])        
        result = transform(r_image)
        st.image(result.permute(1,2,0).numpy(), caption = 'Processed Image.', use_column_width = True)
    except Exception as e:
        print(e)

def fps(image,yolo):
    test_interval = 50
    tact_time = yolo.get_FPS(image, test_interval)
    st.write(str(tact_time) + ' seconds, ', str(1/tact_time),'FPS, @batch_size 1')
    return tact_time
    # print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')


def detect_image(yolo):
    # enable users to upload images for the model to make predictions
    file_up = st.file_uploader("Upload an image", type = ["jpg","png","jpeg"])
    classes = ["up","down","left","right","front","back","clockwise","anticlockwise"]
    class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
    st.sidebar.markdown("See the model preformance and play with it")
    if file_up is not None:
        with st.spinner(text='Preparing Image'):
            # display image that user uploaded
            image = Image.open(file_up)
            st.image(image, caption = 'Uploaded Image.', use_column_width = True)
            st.balloons()
            detect = st.button("开始检测Image")
            if detect:
                st.write("")
                st.write("Just a second ...")
                predict(image,yolo)
                st.balloons()



def detect_camera(yolo):
    picture = st.camera_input("Take a picture")
    if picture:
        filters_to_funcs = {
            "No filter": predict,
            "Heatmap": heatmap,
            "FPS": fps,
        }
        filters = st.selectbox("...and now, apply a filter!", filters_to_funcs.keys())
        image = Image.open(picture)
        with st.spinner(text='Preparing Image'):
            filters_to_funcs[filters](image,yolo)
            st.balloons()

def detect_fps(yolo):
    file_up = st.file_uploader("Upload an image", type = ["jpg","png","jpeg"])
    classes = ["up","down","left","right","front","back","clockwise","anticlockwise"]
    class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
    st.sidebar.markdown("See the model preformance and play with it")
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.balloons()
        detect = st.button("开始检测 FPS")
        if detect:
            with st.spinner(text='Preparing Image'):
                st.write("")
                st.write("Just a second ...")
                tact_time = fps(image,yolo)
                # st.write(str(tact_time) + ' seconds, ', str(1/tact_time),'FPS, @batch_size 1')
                st.balloons()

def heatmap(image,yolo):
    heatmap_save_path = "heatmap_vision.png"
    yolo.detect_heatmap(image, heatmap_save_path)
    img = Image.open(heatmap_save_path)
    transform = transforms.Compose([transforms.ToTensor()])        
    result = transform(img)
    st.image(result.permute(1,2,0).numpy(), caption = 'Processed Image.', use_column_width = True)

def detect_heatmap(yolo):
    file_up = st.file_uploader("Upload an image", type = ["jpg","png","jpeg"])
    classes = ["up","down","left","right","front","back","clockwise","anticlockwise"]
    class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
    st.sidebar.markdown("See the model preformance and play with it")
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.balloons()
        detect = st.button("开始检测 heatmap")
        if detect:
            with st.spinner(text='Preparing Heatmap'):
                st.write("")
                st.write("Just a second ...")
                heatmap(image,yolo)
                st.balloons()

def detect_example(yolo):
    st.sidebar.title("Choose an Image as a example")
    images = os.listdir('./img')
    images.sort()
    image = st.sidebar.selectbox("Image Name", images)
    st.sidebar.markdown("See the model preformance and play with it")
    image = Image.open(os.path.join('img',image))
    st.image(image, caption = 'Choose Image.', use_column_width = True)
    st.balloons()
    detect = st.button("开始检测Image")
    if detect:
        st.write("")
        st.write("Just a second ...")
        predict(image,yolo)
        st.balloons()

def detect_realtime(yolo):

    class VideoProcessor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img = Image.fromarray(img)
            crop            = False
            count           = False
            r_image = yolo.detect_image(img, crop = crop, count=count)
            transform = transforms.Compose([transforms.ToTensor()])        
            result = transform(r_image)
            result = result.permute(1,2,0).numpy()
            result = (result * 255).astype(np.uint8)
            return av.VideoFrame.from_ndarray(result, format="bgr24")
       
    webrtc_ctx = webrtc_streamer(
        key="example",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
        video_processor_factory=VideoProcessor
    )

import cv2
import time
def detect_video(yolo):
    file_up = st.file_uploader("Upload a video", type = ["mp4"])
    print(file_up)
    classes = ["up","down","left","right","front","back","clockwise","anticlockwise"]
    
    if file_up is not None:
        video_path = 'video.mp4'
        st.video(file_up)
        with open(video_path, 'wb') as f:
            f.write(file_up.read())       
        detect = st.button("开始检测 Video")
        
        if detect: 
            video_save_path = 'video2.mp4'
            # display image that user uploaded
            capture = cv2.VideoCapture(video_path)
            
            video_fps = st.slider("Video FPS", 5, 30, int(capture.get(cv2.CAP_PROP_FPS)), 1)
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)


            
            while(True):
                # 读取某一帧
                ref, frame = capture.read()
                if not ref:
                    break
                # 转变成Image
                # frame = Image.fromarray(np.uint8(frame))
                # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                frame = np.array(yolo.detect_image(frame))
                # RGBtoBGR满足opencv显示格式
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                # print("fps= %.2f"%(fps))
                # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
                
            out.release()
            capture.release()
            print("Save processed video to the path :" + video_save_path)
            
            with open(video_save_path, "rb") as file:
                btn = st.download_button(
                        label="Download Video",
                        data=file,
                        file_name="video.mp4",
                    )
            st.balloons()

if __name__ == "__main__":
    main()