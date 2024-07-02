from django.shortcuts import render,redirect

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