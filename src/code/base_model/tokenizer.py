from sympy import sympify, to_dnf
from src.code.base_model.base import BaseTokenizer
import spacy
import subprocess


def get_logical_symbol(token):
    if token == "AND":
        return "&"
    if token == "OR":
        return '|'
    if token == "NOT":
        return '~'

    return token


class Tokenizer(BaseTokenizer):

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.operators = {'&', '|', '~', '(', ')'}

    def tokenize_document(self, document):
        """
        Tokeniza un documento, elimina ruido y stopwords, y devuelve los tokens procesados.

        Args:
            document (str): El documento de texto a ser tokenizado.

        Returns:
            list: Una lista de tokens procesados.
        """
        tokens = self.tokenization(document)
        tokens = self.remove_noise(tokens)
        tokens = self.remove_stopwords(tokens)
        tokens = self.morphological_reduction(tokens)

        return tokens

    def tokenize_query(self, query):
        return self.tokenize_document(query)

    def tokenization(self, document):
        """
        Realiza la tokenización utilizando SpaCy y devuelve una lista de tokens.

        Args:
            document (str): El documento de texto a ser tokenizado.

        Returns:
            list: Una lista de tokens.
        """
        return [token for token in self.nlp(document)]

    def remove_noise(self, tokenized_doc):
        """
        Elimina tokens que no son palabras alfabéticas.

        Args:
            tokenized_doc (list): Una lista de tokens del documento.

        Returns:
            list: Una lista de documentos tokenizados con tokens de ruido eliminados.
        """
        return [token for token in tokenized_doc if token.is_alpha]

    def remove_stopwords(self, tokenized_doc):
        """
        Elimina stopwords del documento tokenizado utilizando la lista de stopwords de SpaCy.

        Args:
            tokenized_doc (list): Una lista de tokens.

        Returns:
            list: Una lista de tokens sin stopwords.
        """
        stopwords = spacy.lang.en.stop_words.STOP_WORDS
        return [
            [token for token in tokenized_doc if token.text not in stopwords]
        ]

    def morphological_reduction(self, tokenized_doc):
        return [
            [token.lemma_ for token in tokenized_doc]
        ]

    def query_to_dnf(self, query):
        """
        Convierte una consulta booleana en forma normal disyuntiva (DNF).

        Args:
            query (str): La consulta booleana en formato de cadena.

        Returns:
            La consulta booleana convertida en DNF.
        """

        tokens = [token.lemma_ for token in self.nlp(query) if
                  token.is_alpha or token.lemma_ in ['(', ')', '&', '|', '~']]

        processed_query = [get_logical_symbol(token.upper()) for token in tokens]

        for i in range(len(processed_query) - 1):
            if processed_query[i] not in self.operators and (processed_query[i + 1] not in self.operators):
                processed_query[i] = processed_query[i] + ' &'

        query_expr = sympify(" ".join(processed_query), evaluate=False)
        query_dnf = to_dnf(query_expr, simplify=True, force=True)

        return query_dnf

    def dnf_to_query(self, query_dnf):
        return [conjunction.replace(' ', '').replace(')', '').replace('(', '').split('&') for conjunction in
                query_dnf.__str__().split('|')]
