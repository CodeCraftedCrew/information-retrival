from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.base_model.base import BaseStorage

class Recommendation:
    
    def __init__(self, storage: BaseStorage):
        self.storage = storage

    def get_recommendations(self, recovered_documents):
        
        """
        Obtiene recomendaciones de documentos basadas en documentos recuperados previamente.

        Args:
            recovered_documents (list): Una lista de índices de documentos recuperados previamente.

        Returns:
            list: Una lista de índices de documentos recomendados.
        """

        # Get all documents from the database
        documents = self.storage.get_all_documents()

        # Initialize TF-IDF vectorizer
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        # Extract titles and genres from documents
        titles = [doc.title for doc in documents]
        genres = [doc.genres.split(", ") for doc in documents]

        # Calculate TF-IDF matrix for titles
        tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

        # Calculate cosine similarity matrix for titles
        cosine_sim_titles = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Initialize dictionaries to store genres counts
        genres_count = {}

        # Iterate over all documents to calculate genres counts
        for doc_genres in genres:
            for genre in doc_genres:
                genres_count[genre] = genres_count.get(genre, 0) + 1

        # Sort genres by count
        sorted_genres = sorted(genres_count.items(), key=lambda x: x[1], reverse=True)

        # Get top genres
        top_genres = [genre[0] for genre in sorted_genres][:5]

        # Initialize recommendations dictionary
        recommendations = {}

        # Iterate over all documents to calculate recommendations based on genres and cosine similarity of titles
        for i, doc in enumerate(documents):
            if doc.id not in recovered_documents:
                # Calculate cosine similarity with documents in results_indices
                sim_sum = 0
                for index in recovered_documents:
                    sim_sum += cosine_sim_titles[i][index]

                # Calculate average cosine similarity
                avg_cosine_sim = sim_sum / len(recovered_documents)

                # Check if any genre of the document is in top genres
                for genre in doc.genres.split(", "):
                    if genre in top_genres:
                        recommendations[doc] = recommendations.get(doc, 0) + genres_count[genre] * avg_cosine_sim

        # Sort recommendations by score
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        # Get recommended indices
        recommended_indices = [rec[0].id for rec in sorted_recommendations]

        # Limit recommendations to 100
        recommended_indices = recommended_indices[:20]

        return recommended_indices