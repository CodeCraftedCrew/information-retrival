from collections import Counter
import gensim
import math


class Vectorizer:

    def build_vocabulary(self, tokenized_docs, no_below=5, no_above=0.5):
        """
        Construye el vocabulario a partir de los documentos tokenizados.

        Args:
            tokenized_docs (list): Una lista de documentos tokenizados.
            no_below (int): Frecuencia mínima de documento para incluir un término en el vocabulario.
            no_above (float): Proporción máxima de documentos para incluir un término en el vocabulario.

        Returns:
            list: Una lista de términos únicos en el vocabulario.
        """

        dictionary = gensim.corpora.Dictionary([[token for token in doc] for (doc, id) in tokenized_docs])
        dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        vocabulary = list(dictionary.token2id.keys())
        return vocabulary

    def calculate_normalized_term_frequency(self, document, tokenized_docs, vocabulary):

        """
        Calcula la frecuencia de términos normalizada para un documento.

        Args:
            document (list): El documento para el cual se calculará la frecuencia de términos normalizada.
            tokenized_docs (list): Una lista de documentos tokenizados.
            vocabulary (list): Vocabulario de términos únicos.

        Returns:
            dict: Un diccionario que mapea términos a su frecuencia normalizada.
        """

        inverse_document_frequency = self.calculate_inverse_document_frequency(
            tokenized_docs, vocabulary
        )

        term_frequency = Counter(document)
        max_inverse_frequency = 0

        for term in document:
            if term in inverse_document_frequency:
                max_inverse_frequency = max(
                    max_inverse_frequency, inverse_document_frequency[term]
                )

        normalized_term_frequency = {
            term: (frequency * (
                inverse_document_frequency[
                    term] if term in inverse_document_frequency else 0)) / max_inverse_frequency
            for term, frequency in term_frequency.items()
        }

        return normalized_term_frequency

    def calculate_inverse_document_frequency(self, tokenized_docs, vocabulary):

        """
        Calcula la frecuencia inversa del documento para los términos en el vocabulario.

        Args:
            tokenized_docs (list): Una lista de documentos tokenizados.
            vocabulary (list): Vocabulario de términos únicos.

        Returns:
            dict: Un diccionario que mapea términos a su frecuencia inversa del documento.
        """

        N = len(tokenized_docs)

        document_frequency = Counter()
        for (doc, id) in tokenized_docs:
            document_frequency.update(set(doc))

        inverse_document_frequency = {
            term: math.log(1 + N / (document_frequency[term] + 1))
            for term in vocabulary
        }

        return inverse_document_frequency
