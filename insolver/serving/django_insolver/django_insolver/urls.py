from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('', include('drf_serving.urls')),
    path('admin/', admin.site.urls),
]
