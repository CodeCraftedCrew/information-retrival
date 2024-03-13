import spacy
from src.code.base_model.base import BaseStorage
import ir_datasets


class MemoryDocumentStorage(BaseStorage):
    def __init__(self, dataset):
        nlp = spacy.load("en_core_web_sm")
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        dataset = ir_datasets.load(dataset)
        texts = [(doc.text, doc.doc_id) for doc in dataset.docs_iter()]
        tokenized_docs = [([token for token in nlp(doc) if token.is_alpha and token.text not in stopwords], id) for (doc, id)
                          in
                          texts]

        reduced_docs = [([token.lemma_ for token in doc], id) for (doc, id) in tokenized_docs]

        self.documents = reduced_docs
        self.documents_raw = [(doc.doc_id, doc.title) for doc in dataset.docs_iter()]

    def save_document(self, document):
        self.documents.append(document)

    def get_all_documents(self):
        return self.documents

    def get_all_raw_documents(self):
        return self.documents_raw
