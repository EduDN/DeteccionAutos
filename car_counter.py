import cv2
import streamlit as st
import numpy as np
import imutils
from PIL import Image
import tempfile
import tensorflow as tf
from imutils.video import FileVideoStream, VideoStream
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Descarga el archivo de pesos y configuración para MobileNet-SSD
SSD_MODEL_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/ssd_mobilenet_v1_coco.pbtxt"
SSD_WEIGHTS_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"

@st.cache(allow_output_mutation=True)
def load_ssd_model():
    config = tf.keras.utils.get_file("ssd_mobilenet_v1_coco.pbtxt", SSD_MODEL_URL, cache_subdir="models")
    weights = tf.keras.utils.get_file("mobilenet_iter_73000.caffemodel", SSD_WEIGHTS_URL, cache_subdir="models")
    model = cv2.dnn.readNetFromCaffe(config, weights)
    return model

class CarCounter(VideoTransformerBase):
    def __init__(self):
        self.model = load_ssd_model()
        self.classes = {7: "Car"}

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        self.model.setInput(blob)
        detections = self.model.forward()

        cars_count = 0
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx in self.classes:
                    cars_count += 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    label = f"{self.classes[idx]}: {confidence * 100:.2f}%"
                    cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"Total Cars: {cars_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return frame

def main():
    st.title("Car Counter using Mobile Camera")
    st.markdown("This app counts cars using your mobile camera and computer vision.")

    webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=CarCounter)

    if webrtc_ctx.video_transformer:
        st.markdown("Car count: **{}**".format(webrtc_ctx.video_transformer.cars_count))

if __name__ == "__main__":
    main()
