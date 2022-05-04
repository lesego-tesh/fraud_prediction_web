from pyexpat import model
from rest_framework.serializers import Serializer, FileField, ListField
from rest_framework import serializers

from Apps.homeApp.models import DataFileUpload,transactionUpload


class UploadSerializer(Serializer):
    id = serializers.IntegerField()
    file_name = serializers.CharField(max_length=50)
    actual_file = FileField()
    description = serializers.CharField(max_length=400)
    
    class Meta:
        model = DataFileUpload
        fields = ['file_name', 'actual_file', 'description']
        read_only_fields = ['id']

class transactionUploadSerializer(Serializer):
    id = serializers.IntegerField()
    file_name = serializers.CharField(max_length=50)
    actual_file = FileField()
    description = serializers.CharField(max_length=400)
    
    class Meta:
        model = transactionUpload
        fields = ['file_name', 'actual_file', 'description']
        read_only_fields = ['id']

# Serializer for multiple files upload.
class MultipleFilesUploadSerializer(Serializer):
    file_name = ListField()
    class Meta:
        model = DataFileUpload
        fields = ['file_name', 'actual_file', 'description']
        read_only_fields = ['id']