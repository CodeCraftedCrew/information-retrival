from sympy import sympify, to_dnf
from core.base_model.base import BaseTokenizer

import nltk
import spacy


class Tokenizer(BaseTokenizer):
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def tokenize(self, document):
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
        
        return tokens
        

    def tokenization(self, document):
        """
        Realiza la tokenización utilizando SpaCy y devuelve una lista de tokens.

        Args:
            document (str): El documento de texto a ser tokenizado.

        Returns:
            list: Una lista de tokens.
        """
        return [token for token in self.nlp(document)]
    
    def remove_noise(self, tokenized_docs):
        """
        Elimina tokens que no son palabras alfabéticas.

        Args:
            tokenized_docs (list): Una lista de documentos tokenizados.

        Returns:
            list: Una lista de documentos tokenizados con tokens de ruido eliminados.
        """
        return [[token for token in doc if token.is_alpha] for doc in tokenized_docs]
    
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
        
    def query_to_dnf(self, query):
        """
        Convierte una consulta booleana en forma normal disyuntiva (DNF).

        Args:
            query (str): La consulta booleana en formato de cadena.

        Returns:
            La consulta booleana convertida en DNF.
        """
    
        tokens = [token.lema_ for token in self.nlp(query) if token.is_alpha]
        
        processed_query = [token.lower().replace("and", "&").replace("or", "|").replace("not", "~") for token in tokens]
        
        query_expr = sympify(processed_query, evaluate=False)
        query_dnf = to_dnf(query_expr, simplify=True)

        return query_dnf
    
    