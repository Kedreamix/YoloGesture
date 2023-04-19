"""Create an Object Detection Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
from yolo import YOLO
import os
import urllib
# 设置网页的icon
st.set_page_config(page_title='Gesture Detector', page_icon='✌',
                   layout='centered', initial_sidebar_state='expanded')

def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(open("instructions.md",encoding='utf-8').read())

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app", "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(open("gesture_streamlit.py",encoding='utf-8').read())
    elif app_mode == "Run the app":
        readme_text.empty()
        run_the_app()

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolotiny_ep100.pth": {
        "url": "https://github.com/Dreaming-future/my_weights/releases/download/v1.0/yolotiny_ep100.pth",
        "size": 23627989
    },
    "yolotiny_SE_ep100.pth": {
        "url": "https://github.com/Dreaming-future/my_weights/releases/download/v1.0/yolotiny_SE_ep100.pth",
        "size": 23802697
    },
    "yolotiny_CBAM_ep100.pth":{
        "url": "https://github.com/Dreaming-future/my_weights/releases/download/v1.0/yolotiny_CBAM_ep100.pth",
        "size": 23978051
    },
    "yolotiny_ECA_ep100.pth":{
        "url": "https://github.com/Dreaming-future/my_weights/releases/download/v1.0/yolotiny_ECA_ep100.pth",
        "size": 23629391
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
        def __init__(self, weights = 'yolotiny_ep100.pth', tiny = True, phi = 0, shape = 416,nms_iou = 0.3, confidence = 0.5):
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
    activities = ["Image", "Video"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    phi = st.sidebar.selectbox("yolov4-tiny 使用的自注意力模式:",('0tiny','1SE','2CABM','3ECA'))
    print("")
    conf,nms = object_detector_ui()
    @st.cache
    def get_yolo(phi,conf,nms):
        weights = 'yolotiny_ep100.pth'
        if phi == '0tiny':
            weights = 'yolotiny_ep100.pth'
        elif phi == '1SE':
            weights = 'yolotiny_SE_ep100.pth'
        elif phi == '2CABM':
            weights = 'yolotiny_CBAM_ep100.pth'
        elif phi == '3ECA':
            weights = 'yolotiny_ECA_ep100.pth'
        opt = Config(weights = weights, tiny = True, phi = int(phi[0]), shape = 416,nms_iou = nms, confidence = conf)
        yolo = YOLO(opt)
        return yolo
    yolo = get_yolo(phi,conf,nms)
    st.write("YOLOV4 tiny 模型加载完毕")
    gesture_detection(yolo)



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
        image = Image.open(image)
        r_image = yolo.detect_image(image, crop = crop, count=count)
        transform = transforms.Compose([transforms.ToTensor()])        
        result = transform(r_image)
        st.image(result.permute(1,2,0).numpy(), caption = 'Processed Image.', use_column_width = True)
    except Exception as e:
        print(e)

def gesture_detection(yolo):
    # enable users to upload images for the model to make predictions
    file_up = st.file_uploader("Upload an image", type = "jpg")
    classes = ["up","down","left","right","front","back","clockwise","anticlockwise"]
    class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
    st.sidebar.markdown("See the model preformance and play with it")
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        st.write("Just a second ...")
        predict(file_up,yolo)

if __name__ == "__main__":
    main()
