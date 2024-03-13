from src.code.base_model.base import BaseHandler, BaseModel, BaseStorage, BaseTokenizer
from src.code.base_model.document import Document
from src.code.base_model.memory_document_storage import MemoryDocumentStorage
from src.code.base_model.tokenizer import Tokenizer


class BooleanHandler(BaseHandler):
    def query(self, documents, query, relaxation_threshold=1):
        """
        Realiza una consulta booleana en los documentos dados.

        Args:
            documents (list): Una lista de documentos.
            query (list): Una lista de términos de consulta.
            relaxation_threshold (float, opcional): Umbral de relajación para la coincidencia de términos.

        Returns:
            list: Una lista de documentos que cumplen con la consulta.
        """
        relevant_documents = [
            (doc, id)
            for (doc, id) in documents
            if self.is_document_relevant(doc, query, relaxation_threshold)
        ]
        return relevant_documents

    def is_document_relevant(self, document, query, relaxation_threshold):
        """
        Verifica si un documento es relevante para una consulta dada.

        Args:
            document (Document): El documento a evaluar.
            query (list): Una lista de términos de consulta.
            relaxation_threshold (int): Umbral de relajación para la coincidencia de términos.

        Returns:
            bool: True si el documento es relevante, False de lo contrario.
        """

        return any(
            [sum(token in document for token in conjunction) >= len(conjunction) * relaxation_threshold for
             conjunction in query])


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


class BooleanModel(BaseModel):
    def __init__(self, storage: BaseStorage = None):
        if storage is None:
            storage = MemoryDocumentStorage()
        handler = BooleanHandler()
        tokenizer = BooleanTokenizer()
        super().__init__(storage, handler, tokenizer)

    def add_document(self, document: Document):
        super().add_document(document)

    def query(self, query, size=None, relaxation_threshold=1):
        """
        Realiza una consulta en el modelo.

        Args:
            query (str): La consulta a realizar.
            size (int, opcional): El tamaño máximo de los documentos recuperados. Si no se proporciona, se devuelven todos los documentos.
            relaxation_threshold (float, opcional): El nivel de relajación de la consulta. Por defecto es 1.

        Returns:
            list: Una lista de documentos relevantes para la consulta.
        """
        processed_query = self.tokenizer.tokenize_query(query)
        if processed_query is None:
            return []
        documents = self.storage.get_all_documents()
        relevant = self.handler.query(documents, processed_query, relaxation_threshold)
        if size is None or size >= len(relevant):
            return [id for _, id in relevant]
        else:
            return [id for _, id in relevant][:size]
