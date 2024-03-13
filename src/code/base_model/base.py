from src.code.base_model.document import Document

class BaseHandler:
    def query(self, documents, query):
        raise NotImplementedError()


class BaseStorage:
    def save_document(self, document):
        raise NotImplementedError()

    def get_all_documents(self):
        raise NotImplementedError()

    def get_all_raw_documents(self):
        raise NotImplementedError()


class BaseTokenizer:
    def tokenize_document(self, document):
        raise NotImplementedError()

    def tokenize_query(self, query):
        raise NotImplementedError()


class BaseModel:

    def __init__(
            self,
            storage: BaseStorage,
            handler: BaseHandler,
            tokenizer: BaseTokenizer,
    ):
        self.storage = storage
        self.handler = handler
        self.tokenizer = tokenizer

    def add_document(self, document: Document):
        self.storage.save_document(document)

    def query(self, query):
        raise NotImplementedError()
