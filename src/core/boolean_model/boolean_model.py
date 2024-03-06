from core.base_model.base import BaseHandler, BaseModel, BaseStorage, BaseTokenizer
from core.base_model.document import Document
from core.base_model.memory_document_storage import MemoryDocumentStorage
from core.base_model.tokenizer import Tokenizer
import numpy as np


class BooleanHandler(BaseHandler):
    def query(self, documents, query, relaxation_threshold=0):
        """
        Realiza una consulta booleana en los documentos dados.

        Args:
            documents (list): Una lista de documentos.
            query (list): Una lista de términos de consulta.
            relaxation_threshold (float, opcional): Umbral de relajación para la coincidencia de términos. Por defecto es 0.

        Returns:
            list: Una lista de documentos que cumplen con la consulta.
        """
        if relaxation_threshold == 0:
            return [
                doc for doc in documents if not any(np.setdiff1d(query, doc.tokens))
            ]
        else:
            min_match = int(len(query) * relaxation_threshold)
            relevant_documents = [
                doc
                for doc in documents
                if self.is_document_relevant(doc, query, min_match)
            ]
            return relevant_documents

    def is_document_relevant(self, document, query, min_match):
        """
        Verifica si un documento es relevante para una consulta dada.

        Args:
            document (Document): El documento a evaluar.
            query (list): Una lista de términos de consulta.
            min_match (int): Número mínimo de términos que deben coincidir para que el documento sea relevante.

        Returns:
            bool: True si el documento es relevante, False de lo contrario.
        """
        matching_terms = sum(term in document.tokens for term in query)
        return matching_terms >= min_match


class BooleanTokenizer(BaseTokenizer):
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
        

class BooleanModel(BaseModel):
    def __init__(self, dataset, storage: BaseStorage = None):
        if storage is None:
            storage = MemoryDocumentStorage()
        handler = BooleanHandler()
        tokenizer = BooleanTokenizer()
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
        relevant = self.handler.query([self.tokenizer.tokenize_document(doc) for doc in documents], processed_query, relaxed)
        if size is None or size >= len(relevant):
            return relevant
        else:
            return relevant[:size]
