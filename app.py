from flask import Flask, request, render_template, jsonify
from PIL import Image as PILImage
import open_clip
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
import zipfile

# Unzip coco_images_resized.zip if not already done
if not os.path.exists("coco_images_resized"):
    with zipfile.ZipFile("coco_images_resized.zip", "r") as zip_ref:
        zip_ref.extractall("coco_images_resized")

# Load precomputed embeddings
with open("image_embeddings.pickle", "rb") as f:
    embeddings_data = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Load the CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Database setup
DATABASE = {
    "images": np.array(embeddings_data["embeddings"]),  # Loaded embeddings
    "image_paths": embeddings_data["image_paths"]  # Corresponding image paths
}

# Helper function to calculate the top 5 most similar images
def get_top_k_similar(query_embedding, db_embeddings, db_paths, top_k=5):
    query_embedding = query_embedding.detach().numpy()
    scores = cosine_similarity(query_embedding, db_embeddings)
    indices = np.argsort(-scores[0])[:top_k]
    return [(db_paths[i], scores[0][i]) for i in indices]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_type = request.form.get("queryType")
        top_k = 5
        results = []

        if query_type == "text":
            # Process text query
            text_query = request.form.get("textQuery")
            text_tokens = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(text_tokens))
            results = get_top_k_similar(text_embedding, DATABASE["images"], DATABASE["image_paths"], top_k)

        elif query_type == "image":
            # Process image query
            image_file = request.files.get("imageQuery")
            if image_file:
                image = PILImage.open(image_file)
                image_tensor = preprocess(image).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image_tensor))
                results = get_top_k_similar(image_embedding, DATABASE["images"], DATABASE["image_paths"], top_k)

        elif query_type == "hybrid":
            # Process hybrid query
            text_query = request.form.get("textQuery")
            image_file = request.files.get("imageQuery")
            weight = float(request.form.get("weight", 0.5))

            if image_file and text_query:
                # Embed text
                text_tokens = tokenizer([text_query])
                text_embedding = F.normalize(model.encode_text(text_tokens))

                # Embed image
                image = PILImage.open(image_file)
                image_tensor = preprocess(image).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image_tensor))

                # Compute weighted hybrid embedding
                hybrid_embedding = F.normalize(weight * text_embedding + (1 - weight) * image_embedding)
                results = get_top_k_similar(hybrid_embedding, DATABASE["images"], DATABASE["image_paths"], top_k)

        # Prepare results with filenames and scores
        results_with_full_path = [
            {"image_path": os.path.join("coco_images_resized", path), "score": score}
            for path, score in results
        ]
        return jsonify({"results": results_with_full_path})

    return render_template("index.html")

# Run the Flask app on port 3000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
