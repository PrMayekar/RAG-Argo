from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/ask/', views.ask_llm, name='ask_llm'),
    path('api/graph/', views.get_graph, name='get_graph'),
    path('api/floats/', views.get_floats, name='get_floats'),
]
