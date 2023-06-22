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

from pipeline.views import get_models, get_result, model_predict, process_dataset
from services.views import (add_dataset, clasify_aes, feedbackPost,
                            fluctuations, get_datasets, get_hypernyms,
                            get_indices, get_jobs, get_languages, get_potential_answers, keywords,
                            keywordsHeatmap, process_cscl, restore_diacritics,
                            ro_correct_text, similar_concepts, syllables)


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
    path('users/me', CurrentUser.as_view()),
    path('services/indices', get_indices),
    path('services/ro-correct-text', ro_correct_text),
    path('services/feedback', feedbackPost),
    path('services/fluctuations', fluctuations),
    path('services/keywords', keywords),
    path('services/keywords-heatmap', keywordsHeatmap),
    path('services/similar-concepts', similar_concepts),
    path('services/hypernyms', get_hypernyms),
    path('services/syllables', syllables),
    path('services/diacritics', restore_diacritics),
    path('services/aes', clasify_aes),
    path('services/cscl', process_cscl),
    path('services/datasets/add', add_dataset),
    path('services/datasets', get_datasets),
    path('services/languages', get_languages),
    path('services/jobs', get_jobs),
    path('services/jobs/<int:job_id>/result', get_result),
    path('services/datasets/<int:dataset_id>/process', process_dataset),
    path('services/qgen/answers', get_potential_answers),
    path('pipeline/models/<int:model_id>/predict', model_predict),
    path('pipeline/models', get_models),
]
