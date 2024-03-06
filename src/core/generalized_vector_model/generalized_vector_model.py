from core.base_model.base import BaseHandler, BaseModel, BaseStorage, BaseTokenizer
from core.base_model.document import Document
from core.base_model.memory_document_storage import MemoryDocumentStorage
from core.base_model.vectorizer import Vectorizer
import numpy as np


class GeneralizedVectorHandler(BaseHandler):
    def query(self, documents, query):
        if query is None:
            return []  # No se pudo vectorizar la consulta

        similarities = [(doc, self.cos_similarity(query, doc)) for doc in documents]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities

    def cos_similarity(self, document_vector, query_vector):
        # Calcula la similitud del coseno entre el vector de consulta y el vector del documento
        dot_product = np.dot(query_vector, document_vector)
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(document_vector)
        if query_norm == 0 or doc_norm == 0:
            return 0  # Evita la división por cero
        else:
            return dot_product / (query_norm * doc_norm)


class GeneralizedVectorTokenizer(BaseTokenizer):
    def __init__(self, vectorizer):
        super().__init__(vectorizer)

    def Tokenize(self, query):
        query.tokens = self.vectorizer.CountTransform(str(query))
        return query.tokens


class GeneralizedVectorModel(BaseModel):
    def __init__(
        self, documents, storage: BaseStorage = None, vectorizer: Vectorizer = None
    ):
        if storage is None:
            storage = MemoryDocumentStorage()
        if vectorizer is None:
            vectorizer = Vectorizer([str(i) for i in documents], True)
        handler = GeneralizedVectorHandler()
        tokenizer = GeneralizedVectorTokenizer(vectorizer)
        super().__init__(documents, storage, vectorizer, handler, tokenizer)

    def add_document(self, document: Document):
        super().add_document(document)

    def query(self, query, size=None):
        processed_query = self.tokenizer.tokenize(query)
        if len(processed_query) == 0:
            return []
        documents = [self.vectorizer.vectorize(doc) for doc in self.storage.get_all_documents()]
        relevant = self.handler.query(documents, processed_query)
        if size is None or size >= len(relevant):
            return relevant
        else:
            return relevant[:size]
