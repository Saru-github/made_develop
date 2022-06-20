
from django.contrib import admin
from django.urls import path, include

from bbs import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('bbs/', include('bbs.urls')),
    path('common/', include('common.urls')),
]
