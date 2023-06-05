from django.urls import path

from . import views
app_name='polls'
urlpatterns = [
    path('', views.first_page),
    path('classification_func/', views.classification_func),
]

