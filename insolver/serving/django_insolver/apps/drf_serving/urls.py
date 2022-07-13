from django.urls import path
from drf_serving.views import *

urlpatterns = [
    path('predict/', PredictAPIView.as_view()),
]
