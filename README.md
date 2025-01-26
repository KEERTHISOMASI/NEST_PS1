# NEST_PS1
Code
Set Up Google Colab
 Create a new notebook:   Go to Google Colab.
        Click on "New Notebook" to create a new project.
 Install Required Libraries:
        Colab already has many libraries pre-installed, but for the ones like spaCy, transformers, and faiss, you may need to install them first.


    !pip install spacy
    !pip install transformers
    !pip install faiss-cpu
    !pip install nltk
    !pip install seaborn
    !pip install scikit-learn

Step 2: Upload Your Data

If you have your dataset ready (e.g., a .csv file from ClinicalTrials.gov), you can upload it directly to Colab using the following method:

    from google.colab import files

    # Upload the dataset
    uploaded = files.upload()

    # After uploading, load the dataset using pandas
    import pandas as pd
    data = pd.read_csv("your_uploaded_file.csv")  # Change the file name accordingly

Step 3: Preprocessing the Data

Once the dataset is loaded, you can apply the preprocessing steps like text cleaning, tokenization, and lemmatization as shown in the code.

 Import necessary libraries:

    import spacy
    import nltk
    from nltk.corpus import stopwords
    from spacy import load

    # Load the spaCy English model
    nlp = load("en_core_web_sm")

    # Ensure that NLTK stopwords are downloaded
    nltk.download("stopwords")

    Define the preprocessing function:

    # Custom function for preprocessing text
    def preprocess_text(text):
    doc = nlp(text.lower())  # Convert text to lowercase and process with spaCy
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords.words("english")]
    return " ".join(tokens)

    # Apply preprocessing to 'Study Title' and 'Brief Summary'
    data['Processed Title'] = data['Study Title'].apply(preprocess_text)
    data['Processed Summary'] = data['Brief Summary'].apply(preprocess_text)

Step 4: Embedding Generation with BioBERT

In Google Colab, you can load BioBERT using the transformers library, which provides pretrained models.
 Import the required model and tokenizer:

    from transformers import AutoTokenizer, AutoModel
    import torch

    # Load BioBERT
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    Function for generating embeddings:

# Function to get embeddings from BioBERT
    def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Apply the embedding function
    data['Title Embedding'] = data['Processed Title'].apply(get_embedding)
    data['Summary Embedding'] = data['Processed Summary'].apply(get_embedding)

Step 5: Clustering with K-Means or DBSCAN

You can use K-Means or DBSCAN to cluster your trials based on the embeddings.

Install and use K-Means for clustering:

    from sklearn.cluster import KMeans
    import numpy as np

# Combine the embeddings
    embeddings = np.vstack(data['Title Embedding'].values)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=10, random_state=42).fit(embeddings)
    data['Cluster'] = kmeans.labels_

Step 6: FAISS for Similarity Search

Now, you can use FAISS to perform fast similarity search.

import faiss

# Build the FAISS index for efficient similarity search
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Query example: You can query using a specific trial or text
      query_embedding = get_embedding("Example clinical trial query")

    # Retrieve top 10 most similar trials
    distances, indices = index.search(query_embedding, k=10)

    # Retrieve the results based on indices
    results = data.iloc[indices[0]]
    print(results[['Study Title', 'Primary Outcome Measure']])

Step 7: Evaluation

You can evaluate the retrieved trials using precision, recall, and F1-score by comparing them to manually labeled relevant trials.

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Example true labels and predictions (replace with actual relevant labels)
    true_labels = [1, 0, 1, 1, 0, 0, 1, 1, 0, 1]  # Ground truth relevance
    predicted_labels = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]  # Predicted relevance

    # Compute metrics
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

Step 8: Final Touches

 Visualizations:
 Visualize clusters or the similarity search using seaborn/matplotlib.

      import seaborn as sns
      import matplotlib.pyplot as plt

# Visualize the cluster results
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=data['Cluster'])
    plt.title("Cluster Visualization")
    plt.show()
