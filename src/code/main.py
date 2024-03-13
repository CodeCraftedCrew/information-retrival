import ir_datasets
from tabulate import tabulate

from src.code.base_model.memory_document_storage import MemoryDocumentStorage
from src.code.base_model.metrics import Metrics
from src.code.boolean_model.boolean_model import BooleanModel
from src.code.boolean_model.extended_boolean_model import ExtendedBooleanModel

dataset = ir_datasets.load("cranfield")

storage = MemoryDocumentStorage("cranfield")
boolean = BooleanModel(storage=storage)
extended = ExtendedBooleanModel(storage=storage)
metrics = Metrics("cranfield")

queries = dataset.queries_iter()

for query_id, query_text in queries:
    print(f"Consulta {query_id}: {query_text}")

    boolean_result = metrics.get_evaluation(query_id, boolean)
    extended_boolean_result = metrics.get_evaluation(query_id, extended)

    rows = [[key, boolean_result[key], extended_boolean_result[key]] for key in boolean_result.keys()]

    print(tabulate(rows, headers=["Metric", "Boolean", "Extended Boolean"], tablefmt="grid"))


    
    