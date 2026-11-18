# Intelligent Fashion Recommendation System Using Machine Learning

# Project Overview
This project develops a fashion recommendation system using Python and machine learning. It compares three content-based recommendation approaches to identify similar fashion products.

The project was implemented in a Kaggle notebook.

# Dataset
The project uses the Fashion Product Images (Small) dataset from Kaggle, containing 44,424 product records.

A reproducible sample of 200 T-shirt products was used for recommendation experiments.

Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small

# Recommendation Models
1. **Metadata-based model:** Compares product colour, gender and usage.
2. **Visual model (ResNet50):** Extracts 2,048-dimensional image features and calculates cosine similarity.
3. **Hybrid model:** Combines 70% visual similarity and 30% metadata similarity.

Each model generates five recommendations for a selected product.

# Experimental Results

| Model | Precision@5 | NDCG@5 |
|---|---:|---:|
| Metadata baseline | 0.800 | 0.830 |
| Visual (ResNet50) | 0.800 | 0.942 |
| Hybrid | 0.900 | 0.870 |

The hybrid model achieved the highest Precision@5, while the visual ResNet50 model achieved the highest NDCG@5.

These findings are preliminary because only two query products had complete manual relevance assessments.

# Technologies Used
- Python
- Kaggle Notebooks
- PyTorch and ResNet50
- Pandas and NumPy
- Scikit-learn
- Matplotlib

## Source Code
The Python implementation is available in the Jupyter notebook (`.ipynb`) included in this repository.
