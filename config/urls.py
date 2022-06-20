
from django.contrib import admin
from django.urls import path, include
from bbs.views import base_views

from bbs import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('bbs/', include('bbs.urls')),
    path('common/', include('common.urls')),
    path('', base_views.index, name ='index'),
]
