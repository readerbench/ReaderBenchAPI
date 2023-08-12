from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _


class RegisterSerializer(serializers.ModelSerializer):

    def validate(self, data):
        model = get_user_model()
        try:
            user = model.objects.get(username=data.get('username'))
            if len(user) > 0:
                raise serializers.ValidationError(_("Username already exists"))
        except model.DoesNotExist:
            pass

        if not data.get('password'):
            raise serializers.ValidationError(_("Empty Password"))

        return data

    class Meta:
        model = get_user_model()
        fields = ('email', 'username', 'first_name', 'last_name', 'password', 'is_active')
        extra_kwargs = {}