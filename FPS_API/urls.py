from django.urls import path, include
from rest_framework import routers
from .views import UploadViewSet, PredictionModelViewSet

router = routers.DefaultRouter()
router.register(r'upload', UploadViewSet, basename="upload")
router.register(r'prediction', PredictionModelViewSet, basename="prediction")

urlpatterns = [
    path('fps/', include(router.urls)),
    path('fps/prediction/<int:id>',PredictionModelViewSet.as_view({'get': 'list'}))
    
]