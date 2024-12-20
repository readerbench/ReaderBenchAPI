"""readerbench URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.contrib.auth import get_user_model
from django.urls import include, path
from oauth2_provider.contrib.rest_framework import TokenHasReadWriteScope
from rest_framework import generics, permissions, serializers

from pipeline.views import (add_dataset, add_indices_job, delete_dataset, delete_model, get_dataset, get_datasets, get_models, get_result,
                            model_feature_importances, model_predict,
                            process_dataset)
from readerbench.view import UserRegister
from services.views import (add_keywords_job, add_offensive_job,
                            add_sentiment_job, delete_job, generate_test,
                            get_job, get_jobs, get_languages,
                            get_potential_answers, process_cscl,
                            restore_diacritics)
from users.views import CustomLoginView, CustomSignUpView, logout_view

# Serializers
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = get_user_model()
        fields = ('username', 'email', "first_name", "last_name")

# API to get current user
class CurrentUser(generics.RetrieveAPIView):
    permission_classes = [permissions.IsAuthenticated, TokenHasReadWriteScope]
    serializer_class = UserSerializer

    def get_object(self):
        return self.request.user

urlpatterns = [
    path('admin/', admin.site.urls),
    path('oauth2/', include('oauth2_provider.urls', namespace='oauth2_provider')),
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    path('accounts/signup/', CustomSignUpView.as_view(), name='signup'),
    path('accounts/logout/', logout_view, name='logout'),
    path('users/me', CurrentUser.as_view()),
    path('services/cscl', process_cscl),
    path('pipeline/datasets/add', add_dataset),
    path('pipeline/datasets', get_datasets),
    path('pipeline/datasets/<int:dataset_id>', get_dataset),
    path('pipeline/datasets/<int:dataset_id>/delete', delete_dataset),
    path('services/languages', get_languages),
    path('services/jobs', get_jobs),
    path('services/jobs/<int:job_id>', get_job),
    path('services/jobs/<int:job_id>/result', get_result),
    path('services/jobs/<int:job_id>/delete', delete_job),
    path('pipeline/datasets/<int:dataset_id>/process', process_dataset),
    path('pipeline/datasets/<int:dataset_id>/indices', add_indices_job),
    path('services/qgen/answers', get_potential_answers),
    path('services/qgen/test', generate_test),
    path('pipeline/models/<int:model_id>/predict', model_predict),
    path('pipeline/models/<int:model_id>/delete', delete_model),
    path('pipeline/models/<int:model_id>/features', model_feature_importances),
    path('pipeline/models', get_models),
    path('services/diacritics', restore_diacritics),
    path('services/keywords', add_keywords_job),
    path('services/sentiment', add_sentiment_job),
    path('services/offensive', add_offensive_job),
]
