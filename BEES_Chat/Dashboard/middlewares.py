from django.shortcuts import render,redirect
from django.utils.deprecation import MiddlewareMixin
import datetime
from django.conf import settings
from django.contrib.auth import logout
#***********Auth Middleware ************
def auth(view_fuction):
    def wrapped_view(request,*args,**kwargs):
        if request.user.is_authenticated==False:
            return redirect('login')
        return view_fuction(request,*args, **kwargs)
    return wrapped_view


#***********Guest User ************
def guest(view_fuction):
    def wrapped_view(request,*args,**kwargs):
        if request.user.is_authenticated:
            return redirect('dashboard')
        return view_fuction(request,*args, **kwargs)
    return wrapped_view


class AutoLogout(MiddlewareMixin):
    def process_request(self, request):
        if not request.user.is_authenticated:
            return

        # Get the last activity time from the session
        last_activity = request.session.get('last_activity')
        now = datetime.datetime.now()

        # If there's no last activity time, set it to now
        if last_activity:
            # Calculate the time difference
            last_activity = datetime.datetime.strptime(last_activity, '%Y-%m-%d %H:%M:%S.%f')
            if (now - last_activity).seconds > settings.AUTO_LOGOUT_DELAY:
                logout(request)
        request.session['last_activity'] = now.strftime('%Y-%m-%d %H:%M:%S.%f')