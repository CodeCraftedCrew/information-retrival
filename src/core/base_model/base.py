from core.base_model.document import Document
import ir_datasets


class BaseHandler:
    def query(self, documents, query):
        raise NotImplementedError()

class BaseStorage:
    def save_document(self, document, representation):
        raise NotImplementedError()

    def get_all_documents(self):
        raise NotImplementedError()

    def get_documents(self, representations):
        raise NotImplementedError()

    def get_documents_rep(self):
        raise NotImplementedError()


class BaseTokenizer:
    def tokenize_query(self):
        raise NotImplementedError()
    
    def tokenize_document(self):
        raise NotImplementedError()


class BaseModel:

    def __init__(
        self,
        dataset,
        storage: BaseStorage,
        handler: BaseHandler,
        tokenizer: BaseTokenizer,
    ):
        self.storage = storage
        self.handler = handler
        self.tokenizer = tokenizer

        dataset = ir_datasets.load(dataset)
        documents = [doc.text for doc in dataset.docs_iter()]

        for doc in documents:
            self.add_document(doc)

    def add_document(self, document: Document):
        representation = self.tokenizer.tokenize(document)
        self.storage.save_document(document, representation)

    def query(self, query):
        raise NotImplementedError()
