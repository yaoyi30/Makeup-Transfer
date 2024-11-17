import time
import cv2
import numpy as np
import onnxruntime

class FaceMakeup:

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

    def split_parse(self, parse):
        h, w = parse.shape
        result = np.zeros([h, w, 10])
        result[:, :, 0][np.where(parse == 0)] = 1
        result[:, :, 0][np.where(parse == 16)] = 1
        result[:, :, 0][np.where(parse == 17)] = 1
        result[:, :, 0][np.where(parse == 18)] = 1
        result[:, :, 0][np.where(parse == 9)] = 1
        result[:, :, 1][np.where(parse == 1)] = 1
        result[:, :, 2][np.where(parse == 2)] = 1
        result[:, :, 2][np.where(parse == 3)] = 1
        result[:, :, 3][np.where(parse == 4)] = 1
        result[:, :, 3][np.where(parse == 5)] = 1
        result[:, :, 1][np.where(parse == 6)] = 1
        result[:, :, 4][np.where(parse == 7)] = 1
        result[:, :, 4][np.where(parse == 8)] = 1
        result[:, :, 5][np.where(parse == 10)] = 1
        result[:, :, 6][np.where(parse == 11)] = 1
        result[:, :, 7][np.where(parse == 12)] = 1
        result[:, :, 8][np.where(parse == 13)] = 1
        result[:, :, 9][np.where(parse == 14)] = 1
        result[:, :, 9][np.where(parse == 15)] = 1
        result = np.array(result)
        return result

    def local_masks(self, split_parse):
        h, w, c = split_parse.shape
        all_mask = np.zeros([h, w])
        all_mask[np.where(split_parse[:, :, 0] == 0)] = 1
        all_mask[np.where(split_parse[:, :, 3] == 1)] = 0
        all_mask[np.where(split_parse[:, :, 6] == 1)] = 0
        all_mask = np.expand_dims(all_mask, axis=2)  # Expansion of the dimension
        all_mask = np.concatenate((all_mask, all_mask, all_mask), axis=2)
        return all_mask

    def prepare_input(self, ori_face_img,ori_parse_img,ref_face_img,ref_parse_img):
        self.ori_face_height, self.ori_face_width = ori_face_img.shape[:2]

        ori_face = ori_face_img.copy()
        ori_parse = ori_parse_img.copy()
        ref_face = ref_face_img.copy()
        ref_parse = ref_parse_img.copy()

        ori_face = cv2.cvtColor(ori_face, cv2.COLOR_BGR2RGB)
        ref_face = cv2.cvtColor(ref_face, cv2.COLOR_BGR2RGB)

        # Resize input image
        ori_face = cv2.resize(ori_face, (self.input_width, self.input_height),interpolation=cv2.INTER_NEAREST)
        ref_face = cv2.resize(ref_face, (self.input_width, self.input_height),interpolation=cv2.INTER_NEAREST)
        ori_parse = cv2.resize(ori_parse, (self.input_width, self.input_height),interpolation=cv2.INTER_NEAREST)
        ref_parse = cv2.resize(ref_parse, (self.input_width, self.input_height),interpolation=cv2.INTER_NEAREST)

        ori_face = ori_face / 127.5 - 1
        ori_face = np.transpose(ori_face, (2, 0, 1))
        ori_face_tensor = ori_face[np.newaxis, :, :, :].astype(np.float32)

        ref_face = ref_face / 127.5 - 1
        ref_face = np.transpose(ref_face, (2, 0, 1))
        ref_face_tensor = ref_face[np.newaxis, :, :, :].astype(np.float32)

        ori_split_parse = self.split_parse(ori_parse)
        ref_split_parse = self.split_parse(ref_parse)

        ori_all_mask = self.local_masks(ori_split_parse)
        ref_all_mask = self.local_masks(ref_split_parse)

        ori_split_parse = np.transpose(ori_split_parse, (2, 0, 1))
        ref_split_parse = np.transpose(ref_split_parse, (2, 0, 1))

        ori_all_mask = np.transpose(ori_all_mask, (2, 0, 1))
        ref_all_mask = np.transpose(ref_all_mask, (2, 0, 1))

        ori_split_parse_tensor = ori_split_parse[np.newaxis, :, :, :].astype(np.float32)
        ref_split_parse_tensor = ref_split_parse[np.newaxis, :, :, :].astype(np.float32)

        ori_all_mask_tensor = ori_all_mask[np.newaxis, :, :, :].astype(np.float32)
        ref_all_mask_tensor = ref_all_mask[np.newaxis, :, :, :].astype(np.float32)

        return ori_face_tensor,ori_split_parse_tensor,ori_all_mask_tensor,ref_face_tensor,ref_split_parse_tensor,ref_all_mask_tensor

    def process_output(self, output):
        predictions = output[2]

        predictions = np.transpose((predictions[0] / 2 + 0.5)*255., (1, 2, 0))
        predictions=np.clip(predictions + 0.5,0,255).astype(np.uint8)

        predictions = cv2.cvtColor(predictions, cv2.COLOR_RGB2BGR)

        predictions = cv2.resize(predictions, (self.ori_face_width, self.ori_face_height),interpolation=cv2.INTER_NEAREST)

        return predictions



    def inference(self,ori_face,ori_parse,ref_face,ref_parse):

        ori_face_tensor, ori_split_parse_tensor, ori_all_mask_tensor, ref_face_tensor, ref_split_parse_tensor, ref_all_mask_tensor = self.prepare_input(
            ori_face, ori_parse, ref_face, ref_parse)

        # Perform inference on the image
        start = time.perf_counter()
        outputs = self.session.run(self.output_names,
                                   {self.input_names[0]: ori_face_tensor, self.input_names[1]: ori_split_parse_tensor,
                                    self.input_names[2]: ori_all_mask_tensor, self.input_names[3]: ref_face_tensor,
                                    self.input_names[4]: ref_split_parse_tensor,
                                    self.input_names[5]: ref_all_mask_tensor})

        print(f"Inference time: {(time.perf_counter() - start) * 1000:.2f} ms")

        self.result = self.process_output(outputs)

        return self.result

