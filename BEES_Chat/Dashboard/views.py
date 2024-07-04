from django.shortcuts import render,redirect,HttpResponse
from django.contrib.auth.forms import UserCreationForm,AuthenticationForm
from django.contrib.auth import login,logout
from .middlewares import auth,guest
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm
from .forms_new import LoginForm, SignUpForm
from datetime import datetime,timedelta
import json
from django.core import serializers
# from django.contrib.auth import get_user_model
import sys
import os
from API.SourceCode.BEES_QA import AzureCosmosQA
from rest_framework.authtoken.models import Token
from azure.cosmos import CosmosClient, exceptions, PartitionKey
# from AdminPanel.models import CustomUser
from django.http import JsonResponse
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

@login_required
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
    query_str = "SELECT * FROM c WHERE 1=1"  # Start with a base query

    # Check if fromDate is provided in the request
    from_date = request.GET.get('fromDate', None)
    if from_date:
        try:
            from_date = datetime.strptime(from_date, '%m/%d/%Y')
             # Adjust to_date to the previous day to cover until the end of the specified day
            from_date = from_date - timedelta(days=1)
            query_str += f" AND c.datetime >= '{from_date.isoformat()}'"
        except ValueError:
            pass

    # Check if toDate is provided in the request
    to_date = request.GET.get('toDate', None)
    if to_date:
        try:
            to_date = datetime.strptime(to_date, '%m/%d/%Y')
            # Adjust to_date to the next day to cover until the end of the specified day
            to_date = to_date + timedelta(days=0)
            query_str += f" AND c.datetime < '{to_date.isoformat()}'"
        except ValueError:
            pass

    print(f"from_date: {from_date}")
    print(f"to_date: {to_date}")
    print(f"query_str: {query_str}")

    # Query the database with the constructed SQL query
    results = list(History_container.query_items(query=query_str, enable_cross_partition_query=True))

    return JsonResponse(results, safe=False)
    
def get_session_details(request):
    
    # session_id=request.body.session_id

    response_data = request.body
    decoded_data = response_data.decode('utf-8')  # Decode bytes to string
    data_dict = json.loads(decoded_data)  # Parse JSON string to dictionary

    session_id = data_dict.get('session_id')
    print(f"Session ID received: {session_id}")

    try:
        query_str = f"SELECT c.responses FROM c WHERE c.session_id = '{session_id}'"
        items = list(History_container.query_items(query=query_str, enable_cross_partition_query=True))
    except Exception as e:
        print(f"Error fetching session details: {e}")
        items = []

    if items:
        
        return JsonResponse(items, safe=False)  # Assuming you want to return the items as a response
    else:
        return HttpResponse({})  # Return an empty dictionary if no items found
    
def logoutFuntion(request):
    logout(request)
    return redirect('login')  