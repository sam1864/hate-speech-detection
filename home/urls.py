from django.contrib import admin
from django.urls import path
from home import views
urlpatterns = [
   path("",views.index,name='home'),
   
   path("text",views.text,name='text'),
   path("micro",views.micro,name='micro'),
   path("link",views.link,name='link'),
   path("video",views.video,name='video'),
   path("image",views.image,name='image'),
   path("txtfile",views.filetotext,name='txtfile'),
   path("history",views.display,name='history'),
   
]
