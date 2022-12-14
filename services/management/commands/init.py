import json
import os

from django.core.management.base import BaseCommand
from rb import Lang

from services.models import Language


class Command(BaseCommand):
    help = 'Initializes DB'

    def handle(self, *args, **options):
        for lang in Lang:
            obj = Language()
            obj.label = lang.name
            obj.save()