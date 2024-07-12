# decorators.py

from django.contrib.auth.decorators import user_passes_test
from django.core.exceptions import PermissionDenied

def admin_required(view_func):
    def wrapped_view(request, *args, **kwargs):
        if request.user.is_authenticated and request.user.groups.filter(name='Admin').exists():
            return view_func(request, *args, **kwargs)
        else:
            raise PermissionDenied
    return wrapped_view
