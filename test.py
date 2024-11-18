import tensorflow as tf
import torch
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


def checkGPU():
    print("Is TensorFlow built with GPU support? ", tf.test.is_built_with_cuda())
    print("TensorFlow GPU devices: ", tf.config.list_physical_devices('GPU'))


def checkOCR(file_extension = ''):
    ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    doc = DocumentFile.from_images(file_extension)
    ocr_result = ocr_model(doc)
    ocr_text = ocr_result.export()
    return ocr_text

def preprocess_ocr_result(ocr_output):
    # Combine all detected text into a single string
    text = " ".join(
        [
            word["value"]
            for page in ocr_output["pages"]
            for block in page["blocks"]
            for line in block.get("lines", [])
            for word in line.get("words", [])
        ]
    )
    return text.strip()

def main():
    checkGPU()
    path_to_image = '/test.py'
    ocr_output = checkOCR(file_extension=path_to_image)
    final_result = preprocess_ocr_result(ocr_output=ocr_output)
    print(final_result)
    
if __name__ == "__main__":
    main()
    
    

