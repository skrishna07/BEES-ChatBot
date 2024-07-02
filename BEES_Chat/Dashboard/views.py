from django.shortcuts import render,redirect
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth import login,logout
from .middlewares import auth,guest
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm
from .forms_new import LoginForm, SignUpForm
# from django.contrib.auth import get_user_model
import sys
import os
from API.SourceCode.BEES_QA import AzureCosmosQA
from rest_framework.authtoken.models import Token
from azure.cosmos import CosmosClient, exceptions, PartitionKey
# from AdminPanel.models import CustomUser

# # Initialize Cosmos Chat History Container:
client = CosmosClient(os.getenv('WebChat_EndPoint'), os.getenv('WebChat_Key'))
database = client.get_database_client(os.getenv('WebChat_DB'))
History_container = database.get_container_client(os.getenv('WebChat_History_Container'))

# User=get_user_model()
# Create your views here.
@guest
def loginPage(request):
    # return render(request,'login.html')
    if  request.method =='POST':
        form=AuthenticationForm(data=request.POST)
        
        print("Form data:", request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request,user)
            return redirect('dashboard')
    else:
        intialData={'username':'', 'password':''}
        form=AuthenticationForm(initial=intialData)
    return render(request, 'login.html',{'form':form})   


def signupPage(request):
    # return render(request,'signup.html')
    if  request.method=='POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print("singup.....")
            print("Form data:", form.cleaned_data)
            # Stop code execution (for debugging purposes)
            # sys.exit()
            user=form.save()
            token = Token.objects.create(user=user)
            login(request,user)
            return redirect('dashboard')
    else:
        intialData={'username':"", 'email':"",'role':"", 'password1':"","password2":""}
        form = UserRegistrationForm(initial=intialData)
    return render(request, 'register.html',{'form':form})    
    
@login_required    
def dashboard(request):
    return render(request,'dashboard.html')  

def getChatHistory(request):
    query_str = f"SELECT * FROM c WHERE c.ip_address = '{ip}'"
    items = list(container.query_items(query=query_str, enable_cross_partition_query=True))
 

def logoutFuntion(request):
    logout(request)
    return redirect('login')  