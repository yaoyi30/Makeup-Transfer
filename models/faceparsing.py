import time
import cv2
import numpy as np
import onnxruntime

class FaceParsing:

    def __init__(self, path):

        # Initialize model
        self.initialize_model(path)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path)
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = image.copy()

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height),interpolation=cv2.INTER_NEAREST)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        input_img = input_img / 255.0
        input_img = input_img - mean
        input_img = input_img / std

        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def process_output(self, output):
        predictions = np.squeeze(output[0]).argmax(0)
        predictions = cv2.resize(np.array(predictions,dtype=np.uint8), (self.img_width, self.img_height),interpolation=cv2.INTER_NEAREST)

        return predictions



    def inference(self,image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")

        self.result = self.process_output(outputs)

        return self.result

