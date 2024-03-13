from django.shortcuts import redirect, render
from django.views import View

from code.base_model.query import Query
from code.boolean_model.boolean_model import BooleanModel
from code.boolean_model.extended_boolean_model import ExtendedBooleanModel

import time

class ModelLoader:
    
    MODELS = {
        "Boolean": BooleanModel,
        "ExtendedBooleanModel": ExtendedBooleanModel,
    }

    def __init__(self):
        self.model = None
        self.modelname = None
        self.dataset = None

    def get_model(self, modelname, dataset):
        if self.modelname == modelname and self.dataset == dataset:
            return self.model

        self.modelname = modelname
        self.dataset = dataset

        model_class = self.MODELS.get(modelname)
        self.model = model_class(dataset) if model_class else None

        return self.model
    
def home(request):
    return render(request, "index.html")

def choose(request):
    modelname = request.POST.get('model')
    dataset = request.POST.get('dataset')
    return redirect('model', modelname=modelname, dataset=dataset)

class ModelView(View):
    template_name = 'search_results.html'

    def get(self, request, modelname, dataset):
        model = ModelLoader().get_model(modelname, dataset)

        size = int(request.GET.get("size", 20))
        query = request.GET.get("query", "")
        data = {"relaxed": False, "iterations": 1, "retroalimentation": False}
        data['query'] = query
        data['size'] = size
        data['model'] = modelname
        data['dataset'] = dataset
        data['time'] = 0

        if query:
            query = Query(id=1, content=query)
            beg_time = time()

            if modelname == "Boolean":
                relaxed_consult = request.GET.get('relaxed') == 'on'
                data['relaxed'] = relaxed_consult
                docs = model.query(query, size=size, relaxed=relaxed_consult)
            elif modelname == "GeneralizedVectorial":
               docs = model.query(query, size=size)

            data['docs'] = docs
            data['time'] = round(time() - beg_time, ndigits=2)

        return render(request, "query-results.html", data)