from django.urls import path, include
from rest_framework import routers

from Apps.homeApp.evaluate import Evaluate
# from .views import UploadViewSet, PredictionModelViewSet
from .views import Evaluation, UploadViewSet,Predictions_upload,predict,predictLatest,Evaluation

router = routers.DefaultRouter()
router.register(r'upload', UploadViewSet, basename="upload")
router.register(r'transactionUpload', Predictions_upload, basename="transactionUpload")
router.register(r'predict', predict, basename="predict")
router.register(r'predictLatest', predictLatest, basename="predictLatest")
router.register(r'evaluation', Evaluation, basename="evaluation")

urlpatterns = [
    path('fps/', include(router.urls)),
    path('fps/predictLatest',predictLatest.as_view({'get': 'list'})),
    path('fps/predict/<int:id>',predict.as_view({'get': 'list'})),
    path('fps/evaluate/<int:id>',Evaluation.as_view({'get': 'list'}))
     
]