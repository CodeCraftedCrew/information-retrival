from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.code.base_model.base import BaseStorage


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

        documents = self.storage.get_all_raw_documents()

        tfidf_vectorizer = TfidfVectorizer(stop_words='english')

        titles = [doc[1] for doc in documents]

        tfidf_matrix = tfidf_vectorizer.fit_transform(titles)

        cosine_sim_titles = cosine_similarity(tfidf_matrix, tfidf_matrix)

        recommendations = {}

        for i, doc in enumerate(documents):
            if doc[0] not in recovered_documents:
                sim_sum = 0
                for index in recovered_documents:
                    sim_sum += cosine_sim_titles[i][int(index)]

                recommendations[doc] = recommendations.get(doc, 0) + sim_sum

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)

        return sorted_recommendations[:5]