from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from oauth2_provider.settings import oauth2_settings
from braces.views import CsrfExemptMixin
from oauth2_provider.views.mixins import OAuthLibMixin
from django.contrib.auth.hashers import make_password

import readerbench.serializers

from django.utils.translation import gettext_lazy as _
from django.db import transaction

class UserRegister(CsrfExemptMixin, OAuthLibMixin, APIView):
    permission_classes = (permissions.AllowAny,)

    server_class = oauth2_settings.OAUTH2_SERVER_CLASS
    validator_class = oauth2_settings.OAUTH2_VALIDATOR_CLASS
    oauthlib_backend_class = oauth2_settings.OAUTH2_BACKEND_CLASS

    def post(self, request):
        if request.auth is None:
            data = request.data
            data['username'] = data.get('email')
            data['password'] = make_password(data.get('password'))     
            data['is_active'] = True
            serializer = readerbench.serializers.RegisterSerializer(data=data)
            if serializer.is_valid():
                try:
                    with transaction.atomic():
                        user = serializer.save()
                        response_data = {
                            'email': 'Account created successfully',
                            'object_data': {
                                'username': user.username,
                            }
                        }
                        return Response(response_data, status=status.HTTP_201_CREATED)
                except Exception as e:
                    return Response(data={"error": e.message}, status=status.HTTP_400_BAD_REQUEST)
            return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        return Response(status=status.HTTP_403_FORBIDDEN) 