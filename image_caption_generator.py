from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def generate_caption(image_path):
    """Generates a caption for the provided image using BLIP."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    img_path = input("Enter the path to your image: ")
    caption = generate_caption(img_path)
    print("\nGenerated Caption:\n", caption)
