
#from django.conf.urls import url
from django.urls import path, include
from .views import (
    csvUploadView,
)

urlpatterns = [

    path('api', csvUploadView.as_view()),

]

