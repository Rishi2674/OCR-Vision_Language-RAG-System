from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from doctr.models import ocr_predictor
from doctr.io import DocumentFile
from PIL import Image
import io
import gc



def convert_pdf_to_image(file_path = ""):
    images = convert_from_path(file_path)
    print(images[0]) #optional
    return images

def MultiModal_RAG(file_path="", query = ""):
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG.index(
    input_path=file_path,
    index_name="image_index", # index will be saved at index_root/index_name/
    store_collection_with_index=False,
    overwrite=True
    )
    text_query = query
    results = RAG.search(text_query, k=1)
    return results

def clear_cache():
    torch.cuda.empty_cache()
    gc.collect()


def OCR(image_index, images):
    ocr_model = ocr_predictor('db_resnet50', 'crnn_vgg16_bn', pretrained=True)
    image = images[image_index]
    document = DocumentFile.from_images(image)
    ocr_result = ocr_model(document)
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
    

def RAG_Pipeline(results,images,query):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                        trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    image_index = results[0]["page_num"] - 1
    image = image[image_index]
    ocr_output = OCR(image_index=image_index,images=images)
    ocr_cleaned_text = preprocess_ocr_result(ocr_output=ocr_output)
    full_text_query = ocr_cleaned_text + " " + query
    new_width, new_height = 256, 256  # Example target size
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    buffer = io.BytesIO()
    resized_image.save(buffer, format="JPEG", quality=75)  # Adjust quality as needed (0-100)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": compressed_image,
            },
            # {"type": "text", "text": text_query +  "Let's think step by step:" }, # With CoT
            {"type": "text", "text": full_text_query + "Answer in single work/phrase: "}, #Without CoT
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text

def main():
    file_path = "\query.pdf"
    query = "What were the four different types of noise considered to examine the robustness of the PRF to noisy data sets"
    images = convert_pdf_to_image(file_path=file_path)
    results = MultiModal_RAG(file_path=file_path,query=query)
    clear_cache()
    final_output = RAG_Pipeline(results=results,images=images,query=query)
    print(final_output)
    
if __name__ == "__main__":
    main()
    


    