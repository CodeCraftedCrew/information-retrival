from tabulate import tabulate
import numpy as np

from src.code.base_model.base import BaseHandler, BaseModel, BaseStorage, BaseTokenizer
from src.code.base_model.document import Document
from src.code.base_model.memory_document_storage import MemoryDocumentStorage
from src.code.base_model.recommendation import Recommendation
from src.code.base_model.tokenizer import Tokenizer
from src.code.base_model.vectorizer import Vectorizer


class ExtendedBooleanHandler(BaseHandler):
    def query(self, documents, query, p=1, relevance_threshold=0.5):
        """
        Realiza una consulta extendida en los documentos dados.

        Args:
            documents (list): Una lista de documentos.
            query (list): Una lista de términos de consulta.
            p (int, opcional): El valor de p para la métrica de similitud. Por defecto es 1.
            relevance_threshold (float, opcional): Umbral de relevancia para los documentos recuperados. Por defecto es 0.5.

        Returns:
            list: Una lista de documentos que cumplen con la consulta extendida.
        """

        vectorizer = Vectorizer()
        vocabulary = vectorizer.build_vocabulary(tokenized_docs=documents)
        relevant_documents_with_scores = [
            (doc, id, relevance_score)
            for (doc, id, relevance_score) in [
                (doc, id, self.is_document_relevant(
                    vectorizer.calculate_normalized_term_frequency(
                        document=doc,
                        tokenized_docs=documents, vocabulary=vocabulary
                    ),
                    query,
                    p
                ))
                for (doc, id) in documents if len(doc) > 0
            ]
        ]

        relevant_documents_with_scores.sort(key=lambda x: x[2], reverse=True)
        return [(doc, id) for (doc, id, score) in relevant_documents_with_scores if score >= relevance_threshold]



    def is_document_relevant(self, weights, query, p):
        """
        Verifica si un documento es relevante para una consulta extendida dada.

        Args:
            weights (dict): Un diccionario que contiene los pesos de los términos.
            query (list): Una lista de términos de consulta.
            p (int): El valor de p para la métrica de similitud.

        Returns:
            bool: True si el documento es relevante, False de lo contrario.
        """

        weights_query_terms = [
            [weights.get(term.lower(), 0.0) for term in fnd]
            for fnd in query
        ]
        weights_or = [self.and_similarity(terms, p) for terms in weights_query_terms]

        return self.or_similarity(weights_or, p)

    def or_similarity(self, weights, p):
        """
        Calcula la similitud de la consulta disyuntiva.

        Args:
            weights (list): Una lista de pesos de los términos.
            p (int): El valor de p para la métrica de similitud.
            t (int): Total de terminos en el documento

        Returns:
            float: La similitud OR calculada.
        """
        return ((sum([weight ** p for weight in weights])) / len(weights)) ** (1 / p)

    def and_similarity(self, weights, p):
        """
        Calcula la similitud de la consulta conjuntiva.

        Args:
            weights (list): Una lista de pesos de los términos.
            p (int): El valor de p para la métrica de similitud.
            t (int): Total de terminos en el documento

        Returns:
            float: La similitud AND calculada.
        """
        return 1 - ((sum([(1 - weight) ** p for weight in weights])) / len(weights)) ** (1 / p)


class ExtendedBooleanTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__()

    def tokenize_query(self, query):
        """
        Tokeniza una consulta y la convierte en forma normal disyuntiva (DNF).

        Args:
            query (str): La consulta a tokenizar.

        Returns:
            La consulta en forma normal disyuntiva (DNF).
        """
        tokenizer = Tokenizer()
        dnf = tokenizer.query_to_dnf(query)
        query = tokenizer.dnf_to_query(dnf)
        return query

    def tokenize_document(self, document):
        """
        Tokeniza un documento.

        Args:
            document (str): El documento a tokenizar.

        Returns:
            list: Una lista de tokens del documento.
        """

        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize_document(document)
        return tokens


class ExtendedBooleanModel(BaseModel):
    def __init__(self, storage: BaseStorage = None):
        if storage is None:
            storage = MemoryDocumentStorage()
        handler = ExtendedBooleanHandler()
        tokenizer = ExtendedBooleanTokenizer()
        super().__init__(storage, handler, tokenizer)

    def add_document(self, document: Document):
        super().add_document(document)

    def query(self, query, size=None):
        """
        Realiza una consulta en el modelo.

        Args:
            query (str): La consulta a realizar.
            size (int, opcional): El tamaño máximo de los documentos recuperados. Si no se proporciona, se devuelven todos los documentos.

        Returns:
            list: Una lista de documentos relevantes para la consulta.
        """
        processed_query = self.tokenizer.tokenize_query(query)
        if len(processed_query) == 0:
            return []
        documents = self.storage.get_all_documents()
        relevant = self.handler.query(documents,
                                      processed_query)

        print(tabulate([(id, " ".join([token for token in doc[:20]])) for (doc, id) in relevant][:5],
                       headers=["Id", "Start"], tablefmt="grid"))

        ids = [id for (doc, id) in relevant]

        recommended = Recommendation(self.storage).get_recommendations(ids)

        print(tabulate(recommended, headers=["Id", "Title"], tablefmt="grid"))

        return ids if size is None or size >= len(relevant) else ids[:size]
