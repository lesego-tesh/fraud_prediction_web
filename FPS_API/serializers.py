from rest_framework import serializers
from Apps.homeApp.models import DataFileUpload

class fpsSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataFileUpload
        fields = [" file_name", "actual_file", "description "]
