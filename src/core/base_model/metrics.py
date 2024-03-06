from core.base_model.base import BaseModel
import ir_datasets

class Metrics:
    
    def __init__(self, dataset):
        dataset = ir_datasets.load(dataset)
        
    def relevant_documents(self, query_id : str):
        """
        Obtiene los documentos relevantes para una consulta específica.

        Args:
            query_id (str): El identificador único de la consulta.

        Returns:
            tuple: Una tupla que contiene una lista de identificadores de documentos relevantes y el texto de la consulta.
        """
        
        for (queryt_id, query_text) in self.dataset.queries_iter():
            if queryt_id == query_id:
                break
            
        return (
            [
            doc_id
            for (queryt_id, doc_id, relevance, iteration) in self.dataset.qrels_iter()
            if queryt_id == query_id and relevance in [3, 4]
            ], 
            query_text)
    
    def precision(self, recovered_documents, relevant_documents):
        """
        Calcula la precisión dada una lista de documentos recuperados y documentos relevantes.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.

        Returns:
            float: La precisión calculada.
        """
        return len(set(recovered_documents) & set(relevant_documents)) / len(recovered_documents)
    
    def recall(self, recovered_documents, relevant_documents):
        """
        Calcula la recuperación dada una lista de documentos recuperados y documentos relevantes.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.

        Returns:
            float: La recuperación calculada.
        """
        return len(set(recovered_documents) & set(relevant_documents)) / len(relevant_documents)
    
    def f(self, recovered_documents, relevant_documents, beta):
        """
        Calcula la medida Fβ dada una lista de documentos recuperados, documentos relevantes y un valor β.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.
            beta (float): El valor de β para la medida Fβ.

        Returns:
            float: La medida Fβ calculada.
        """
        return (1 + beta ** 2) * self.precision(recovered_documents, relevant_documents) * self.recall(recovered_documents, relevant_documents) / (beta ** 2 * self.precision(recovered_documents, relevant_documents) + self.recall(recovered_documents, relevant_documents))
    
    def f1(self, recovered_documents, relevant_documents):
        """
        Calcula la medida F1 dada una lista de documentos recuperados y documentos relevantes.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.

        Returns:
            float: La medida F1 calculada.
        """
        return self.f(recovered_documents, relevant_documents, 1)
    
    def r_precicion(self, recovered_documents, relevant_documents, r):
        """
        Calcula la precisión en r dados una lista de documentos recuperados, documentos relevantes y un valor r.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.
            r (int): El valor r para calcular la precisión en r.

        Returns:
            float: La precisión en r calculada.
        """
        return self.precision(recovered_documents[:r], relevant_documents)
    
    def fallout(self, recovered_documents, relevant_documents):
        """
        Calcula el Fallout dado una lista de documentos recuperados y documentos relevantes.

        Args:
            recovered_documents (list): Una lista de identificadores de documentos recuperados.
            relevant_documents (list): Una lista de identificadores de documentos relevantes.

        Returns:
            float: El Fallout calculado.
        """
        fp = len(set(recovered_documents).difference(set(relevant_documents)))
        tn = len(set(self.dataset.docs_iter().map(lambda x: x[0])).difference(set(recovered_documents).union(set(relevant_documents))))
        return fp / (fp + tn)
    
    def get_evaluation(self, query_id : str, model: BaseModel):
        """
        Obtiene la evaluación de un modelo dado una consulta específica.

        Args:
            query_id (str): El identificador único de la consulta.
            model (BaseModel): El modelo de recuperación de información.

        Returns:
            dict: Un diccionario que contiene las métricas de evaluación calculadas (precisión, recuperación, F1, precisión en r y Fallout).
        """
        relevant_documents, query_text = self.relevant_documents(query_id)
        recovered_documents = model.query(query_text)
        
        return {
            "precision": self.precision(recovered_documents, relevant_documents),
            "recall": self.recall(recovered_documents, relevant_documents),
            "f1": self.f1(recovered_documents, relevant_documents),
            "r-precision": self.r_precicion(recovered_documents, relevant_documents, len(relevant_documents)),
            "fallout": self.fallout(recovered_documents, relevant_documents)
        }