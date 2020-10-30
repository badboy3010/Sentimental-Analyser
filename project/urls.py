from django.conf.urls import  url
from . import views
urlpatterns = [

    url(r'^$', views.index, name='index'),
    url(r'^values/$', views.values, name='value'),
    url(r'^about/$', views.about, name='about'),

]