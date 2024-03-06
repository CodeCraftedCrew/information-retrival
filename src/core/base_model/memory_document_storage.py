from core.base_model.base import BaseStorage
from numpy import array_equal

class MemoryDocumentStorage(BaseStorage):
    def __init__(self):
        self.documents = []

    def save_document(self, document, representation):
        self.documents.append(document)

    def get_all_documents(self):
        return self.documents

    def get_documents(self, representations):
        documents = []
        for doc in documents:
            if any(r for r in representations if array_equal(doc.tokens, r)):
                documents.append(doc)
        return documents

    def get_documents_rep(self):
        return (d.tokens for d in self.documents)