from core.base_model.base import BaseHandler, BaseModel, BaseStorage, BaseTokenizer
from core.base_model.document import Document
from core.base_model.memory_document_storage import MemoryDocumentStorage
from core.base_model.tokenizer import Tokenizer
from core.base_model.vectorizer import Vectorizer


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
        relevant_documents = [
            doc
            for doc in documents
            if self.is_document_relevant(
                vectorizer.calculate_normalized_term_frequency(
                    documents, vectorizer.build_vocabulary(documents)
                ),
                query,
                p,
                relevance_threshold,
            )
        ]
        return relevant_documents

    def is_document_relevant(self, weights, query, p, relevance_threshold):
        
        """
        Verifica si un documento es relevante para una consulta extendida dada.

        Args:
            weights (dict): Un diccionario que contiene los pesos de los términos.
            query (list): Una lista de términos de consulta.
            p (int): El valor de p para la métrica de similitud.
            relevance_threshold (float): Umbral de relevancia para los documentos recuperados.

        Returns:
            bool: True si el documento es relevante, False de lo contrario.
        """

        weights_query_terms = [[weights[term] for term in fnd] for fnd in query]

        weights_or = sum(self.and_similarity(terms, p) for terms in weights_query_terms)

        return self.or_similarity(weights_or, p) >= relevance_threshold

    def or_similarity(self, weights, p):
        """
        Calcula la similitud de la consulta disyuntiva.

        Args:
            weights (list): Una lista de pesos de los términos.
            p (int): El valor de p para la métrica de similitud.

        Returns:
            float: La similitud OR calculada.
        """
        return sum([weight**p for weight in weights]) ** (1 / p)

    def and_similarity(self, weights, p):
        """
        Calcula la similitud de la consulta conjuntiva.

        Args:
            weights (list): Una lista de pesos de los términos.
            p (int): El valor de p para la métrica de similitud.

        Returns:
            float: La similitud AND calculada.
        """
        return sum([(1 - weight) ** p for weight in weights]) ** (1 / p)


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
        tokens = tokenizer.tokenize(query)
        dnf = tokenizer.query_to_dnf(tokens)
        return dnf

    def tokenize_document(self, document):
        
        """
        Tokeniza un documento.

        Args:
            document (str): El documento a tokenizar.

        Returns:
            list: Una lista de tokens del documento.
        """
        
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(document)
        return tokens


class ExtendedBooleanModel(BaseModel):
    def __init__(self, dataset, storage: BaseStorage = None):
        if storage is None:
            storage = MemoryDocumentStorage()
        handler = ExtendedBooleanHandler()
        tokenizer = ExtendedBooleanTokenizer()
        super().__init__(dataset, storage, handler, tokenizer)

    def add_document(self, document: Document):
        super().add_document(document)

    def query(self, query, size=None, relaxed=0):
        """
        Realiza una consulta en el modelo.

        Args:
            query (str): La consulta a realizar.
            size (int, opcional): El tamaño máximo de los documentos recuperados. Si no se proporciona, se devuelven todos los documentos.
            relaxed (float, opcional): El nivel de relajación de la consulta. Por defecto es 0.

        Returns:
            list: Una lista de documentos relevantes para la consulta.
        """
        processed_query = self.tokenizer.tokenize_query(query)
        if len(processed_query) == 0:
            return []
        documents = self.storage.get_all_documents()
        relevant = self.handler.query(
            [self.tokenizer.tokenize_document(doc) for doc in documents],
            processed_query,
            relaxed,
        )
        if size is None or size >= len(relevant):
            return relevant
        else:
            return relevant[:size]
