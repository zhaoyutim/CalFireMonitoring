from django.http import HttpResponse

from django.views.decorators.http import require_http_methods


@require_http_methods(["GET"])
def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")
