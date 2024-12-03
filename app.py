from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image as PILImage
import open_clip
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os

# Load precomputed embeddings from pickle file
df = pd.read_pickle("image_embeddings.pickle")

# Convert DataFrame to database format
DATABASE = {
    "images": np.vstack(df["embedding"].values),  # Stack the embeddings into a NumPy array
    "image_paths": df["file_name"].tolist()  # Convert file names to a list
}

# Initialize the Flask app
app = Flask(__name__)

# Load the CLIP model and tokenizer
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.eval()

# Helper function to calculate the top 5 most similar images
def get_top_k_similar(query_embedding, db_embeddings, db_paths, top_k=5):
    if not isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.detach().numpy()  # Only detach if it's a tensor
    scores = cosine_similarity(query_embedding, db_embeddings)
    indices = np.argsort(-scores[0])[:top_k]
    return [(db_paths[i], scores[0][i]) for i in indices]


# Function to apply PCA dimensionality reduction
def apply_pca(embeddings, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('coco_images_resized', filename)


@app.route('/', methods=['GET', 'POST'])
def index():
    results_with_full_path = []
    if request.method == 'POST':
        query_type = request.form.get("queryType")
        top_k = 5
        use_pca = request.form.get("usePCA") == "on"
        n_components = int(request.form.get("kComponents", 10)) if use_pca else None
        results = []

        if use_pca:
            # Apply PCA to database embeddings
            pca = PCA(n_components=n_components)
            db_embeddings = pca.fit_transform(DATABASE["images"])
        else:
            db_embeddings = DATABASE["images"]

        if query_type == "text":
            text_query = request.form.get("textQuery")
            text_tokens = tokenizer([text_query])
            text_embedding = F.normalize(model.encode_text(text_tokens)).detach().numpy()  # Detach and convert to NumPy

            if use_pca:
                text_embedding = pca.transform(text_embedding)

            results = get_top_k_similar(text_embedding, db_embeddings, DATABASE["image_paths"], top_k)

        elif query_type == "image":
            image_file = request.files.get("imageQuery")
            if image_file:
                image = PILImage.open(image_file)
                image_tensor = preprocess(image).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image_tensor)).detach().numpy()  # Detach and convert to NumPy


                if use_pca:
                    image_embedding = pca.transform(image_embedding)

                results = get_top_k_similar(image_embedding, db_embeddings, DATABASE["image_paths"], top_k)

        elif query_type == "hybrid":
            text_query = request.form.get("textQuery")
            image_file = request.files.get("imageQuery")
            weight = float(request.form.get("weight", 0.5))

            if image_file and text_query:
                text_tokens = tokenizer([text_query])
                text_embedding = F.normalize(model.encode_text(text_tokens)).detach().numpy()  # Detach and convert to NumPy

                image = PILImage.open(image_file)
                image_tensor = preprocess(image).unsqueeze(0)
                image_embedding = F.normalize(model.encode_image(image_tensor)).detach().numpy()  # Detach and convert to NumPy

                hybrid_embedding = weight * text_embedding + (1 - weight) * image_embedding

                if use_pca:
                    hybrid_embedding = pca.transform(hybrid_embedding)

                results = get_top_k_similar(hybrid_embedding, db_embeddings, DATABASE["image_paths"], top_k)

        # Prepare results with filenames and scores
        results_with_full_path = [
            {"image_path": f"/images/{path}", "score": float(score)}
            for path, score in results
        ]


    return render_template("index.html", results=results_with_full_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
