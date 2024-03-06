from django.urls import path

from . import views

app_name = "documents"
urlpatterns = [
    path("", views.home, name="home"),
    path("model/<str:model_name>/<str:data_set>", views.ModelView.as_view(), name="model"),
    path("choose", views.choose, name="choose"),
]