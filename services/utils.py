from rb import Lang

from services.models import Language


def init_languages(apps, schema_editor):
    for lang in Lang:
        obj = Language()
        obj.label = lang.name
        obj.save()