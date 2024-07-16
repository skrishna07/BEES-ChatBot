from django.shortcuts import render,redirect,HttpResponse,get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login,logout,authenticate
from .middlewares import auth,guest
from django.contrib.auth.models import Group
from django.contrib.auth.decorators import login_required
from .forms import UserRegistrationForm
from datetime import datetime,timedelta
from collections import defaultdict
import json
import time
import django.views.decorators.csrf
# from django.contrib.auth import get_user_model
import os
from .models import User
from API.SourceCode.BEES_QA import AzureCosmosQA
from rest_framework.authtoken.models import Token
from azure.cosmos import CosmosClient, exceptions, PartitionKey
# from AdminPanel.models import CustomUser
from django.http import JsonResponse,HttpResponseBadRequest
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger

# # Initialize Cosmos Chat History Container:
client = CosmosClient(os.getenv('WebChat_EndPoint'), os.getenv('WebChat_Key'))
database = client.get_database_client(os.getenv('WebChat_DB'))
History_container = database.get_container_client(os.getenv('WebChat_History_Container'))

# User=get_user_model()
# Create your views here.
@guest
def loginPage(request):
    
    if  request.method =='POST':
        form=AuthenticationForm(data=request.POST)
        
        if form.is_valid():
            user = form.get_user()
            login(request,user)
            return redirect('dashboard')
    else:
        intialData={'username':'', 'password':''}
        form=AuthenticationForm(initial=intialData)
    return render(request, 'login.html',{'form':form})   

@login_required
@django.views.decorators.csrf.csrf_exempt
def signupPage(request):
    if  request.method=='POST':
        user_id = request.POST.get('userId', None)
        
        if user_id:
            # Update user details
            try:
                user = get_object_or_404(User, id=user_id)
                user.username = request.POST.get('username')
                user.email = request.POST.get('email')
                user.role = request.POST.get('role')  # Assuming 'role' is a field in a related 'Profile' model
                user.is_active = request.POST.get('status') == 'true'  # Assuming 'status' is a boolean field
                user.save()

                user.groups.clear()
                if user.role == 'admin':
                    group = Group.objects.get(name='Admin')
                elif user.role == 'user':
                    group = Group.objects.get(name='User')

                user.groups.add(group)
                return JsonResponse({'status': 200, 'message': 'User updated successfully'})
            except User.DoesNotExist:
                return HttpResponseBadRequest("User does not exist")
            except Exception as e:
                return HttpResponseBadRequest(f"An error occurred: {str(e)}")

        else:
            form = UserRegistrationForm(request.POST)
            if form.is_valid():
                
                # Stop code execution (for debugging purposes)
                # sys.exit()
                user=form.save()
                # Assign user to a specific group based on role
                if user.role == 'admin':
                    group = Group.objects.get(name='Admin')
                elif user.role == 'user':
                    group = Group.objects.get(name='User')

                user.groups.add(group)

                token = Token.objects.create(user=user)
                return JsonResponse({'status': 200, 'message': 'User created successfully'})
            else:
                errors = form.errors
                return JsonResponse({'status': 400, 'errors': errors})
            
    else:
        if request.user.is_authenticated:
        # Fetch all groups associated with the user
            user_groups = request.user.groups.all()
            user_is_admin = user_groups.filter(name='Admin').exists()
            # Pass user_is_admin and other necessary data to the template
            intialData={'username':"", 'email':"",'role':"", 'password1':"","password2":""}
            form = UserRegistrationForm(initial=intialData)

            return render(request, 'user_management.html', {'form':form,'user_is_admin': user_is_admin})
        else:
            # Handle the case where the user is not authenticated
            # Redirect to login or handle appropriately
            return HttpResponse("You are not authenticated.")
    
@login_required    
def dashboard(request):
    if request.user.is_authenticated:
        # Fetch all groups associated with the user
        user_groups = request.user.groups.all()
        user_is_admin = user_groups.filter(name='Admin').exists()

        today = datetime.now().date().strftime('%m/%d/%Y')
        seven_days_ago = (datetime.now().date() - timedelta(days=6  )).strftime('%m/%d/%Y')
        dates = {
            'today': today,
            'seven_days_ago': seven_days_ago
        }

        # Pass user_is_admin and other necessary data to the template
        return render(request, 'dashboard.html', {'user_is_admin': user_is_admin,'dates':dates})
    else:
        # Handle the case where the user is not authenticated
        # Redirect to login or handle appropriately
        return HttpResponse("You are not authenticated.")
    

@login_required    
def userEngagement(request):
    if request.user.is_authenticated:
        # Fetch all groups associated with the user
        user_groups = request.user.groups.all()
        user_is_admin = user_groups.filter(name='Admin').exists()

        today = datetime.now().date().strftime('%m/%d/%Y')
        seven_days_ago = (datetime.now().date() - timedelta(days=6)).strftime('%m/%d/%Y')
        dates = {
            'today': today,
            'seven_days_ago': seven_days_ago
        }
        # Pass user_is_admin and other necessary data to the template
        return render(request, 'user_engagement.html', {'user_is_admin': user_is_admin,'dates':dates})
    else:
        # Handle the case where the user is not authenticated
        # Redirect to login or handle appropriately
        return HttpResponse("You are not authenticated.")

@login_required    
def sessionAnalytics(request):
    if request.user.is_authenticated:
        # Fetch all groups associated with the user
        user_groups = request.user.groups.all()
        user_is_admin = user_groups.filter(name='Admin').exists()

        today = datetime.now().date().strftime('%m/%d/%Y')
        seven_days_ago = (datetime.now().date() - timedelta(days=6)).strftime('%m/%d/%Y')
        dates = {
            'today': today,
            'seven_days_ago': seven_days_ago
        }
        # Pass user_is_admin and other necessary data to the template
        return render(request, 'session_analytics.html', {'user_is_admin': user_is_admin,'dates':dates})
    else:
        # Handle the case where the user is not authenticated
        # Redirect to login or handle appropriately
        return HttpResponse("You are not authenticated.")


@django.views.decorators.csrf.csrf_exempt
def getChatHistory(request):
    try:
        query_str = "SELECT * FROM c WHERE 1=1"  # Start with a base query
        # Get today's date and the current month
        today = datetime.today().date()
        current_month = today.strftime('%Y-%m')

        # Check if fromDate is provided in the request
        from_date_str = request.GET.get('fromDate', None)
        
        if from_date_str:
            try:
                from_date = datetime.strptime(from_date_str, '%m/%d/%Y')
                # Adjust from_date to include the start of the day
                from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)- timedelta(days=1)
                query_str += f" AND c.datetime >= '{from_date.isoformat()}'"
            except ValueError:
                pass
        
        # Check if toDate is provided in the request
        to_date_str = request.GET.get('toDate', None)
        if to_date_str:
            try:
                to_date = datetime.strptime(to_date_str, '%m/%d/%Y')
                current_month = to_date.strftime('%Y-%m')
                # Adjust to_date to include the start of the next day
                to_date = to_date.replace(hour=0, minute=0, second=0, microsecond=0)
                query_str += f" AND c.datetime < '{to_date.isoformat()}'"
            except ValueError:
                pass
        
        # If from_date is not set, set it to 7 days before the current date
        if not from_date_str:
            from_date = datetime.now() - timedelta(days=7)
            from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
            query_str += f" AND c.datetime >= '{from_date.isoformat()}'"

        # If to_date is not set, set it to the current date
        if not to_date_str:
            to_date = datetime.now()
            to_date = to_date.replace(hour=0, minute=0, second=0, microsecond=0)
            query_str += f" AND c.datetime < '{to_date.isoformat()}'"

        # Check if search value is provided in the request
        search = request.POST.get('search', None)
        if search:
            # Add search functionality for ip_address and session_id
            query_str += f" AND (CONTAINS(c.ip_address, '{search}') OR CONTAINS(c.session_id, '{search}'))"

        # Query the database with the constructed SQL query
        results = list(History_container.query_items(query=query_str, enable_cross_partition_query=True))

        # Construct the query to get unique IPs for the current month
        ip_query_str = f"""
        SELECT DISTINCT c.ip_address FROM c 
        WHERE STARTSWITH(c.datetime, '{current_month}')
        """
        
        # Execute the query
        ip_results = list(History_container.query_items(query=ip_query_str, enable_cross_partition_query=True))

        # Print or process the results
        distict_ips = [item['ip_address'] for item in ip_results]
        
        if not results:
            return JsonResponse({
                'draw': request.POST.get('draw'),
                'recordsTotal': 0,
                'recordsFiltered': 0,
                'data': [],
                'daily_unique_ips': {},
                'monthly_unique_ips': {},
                'unique_ips_today': 0,
                'unique_ips_current_month': 0,
                'frequency_of_use': 0,
                'average_token_cost_per_user': 0,
                'average_tokens_per_ip':0,
                'average_tokens_per_session':0,
                'average_sessions_per_ip':0,
                'user_retention_rate': 0
            })

            # Sort the results by the 'datetime' field in descending order
        sorted_results = sorted(results, key=lambda x: datetime.strptime(x['datetime'], '%Y-%m-%d %H:%M:%S.%f'), reverse=True)
        
        start_time = time.time()
        draw = request.POST.get('draw')
        start = int(request.POST.get('start', 0))
        length = int(request.POST.get('length', 10))
        
        # Paginate the data
        if length == -1:
            length = len(sorted_results)
        paginator = Paginator(sorted_results, length)
        page_number = start // length + 1

        try:
            data = paginator.page(page_number)
        except PageNotAnInteger:
            data = paginator.page(1)
        except EmptyPage:
            data = []

        result = []

        # Track unique session IDs
        unique_sessions = set()
        daily_unique_ips = defaultdict(set)
        monthly_unique_ips = defaultdict(set)
        
        # Track unique IPs for today and the current month
        unique_ips_today = set()
        unique_ips_current_month = set()

        # Initialize variables
        sessions_by_ip = defaultdict(set)
        unique_ips = set()
        first_access_by_ip = {}

        # Initialize counters
        daily_unique_users_count = 0
        total_tokens_used = 0
        total_unique_ips = 0
        total_token_cost = 0.0
        total_sessions = 0
        
        returning_ips = set()

        for row in sorted_results:
            # Parse the string into a datetime object
            datetime_str = row['datetime']
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')

            ip_address = row['ip_address']
            # Extract the date and time
            date_part = dt.date()
            time_part = dt.time()
            month_part = dt.strftime('%Y-%m')

            # Track unique sessions
            unique_sessions.add(row['session_id'])

            # Track unique users daily
            if ip_address and date_part == today:
                unique_ips_today.add(ip_address)

            # Track monthly unique users
            if ip_address:
                monthly_unique_ips[month_part].add(ip_address)

             # Track unique IPs for today and the current month
            if date_part == today and row['ip_address']:
                unique_ips_today.add(row['ip_address'])
            if month_part == current_month and row['ip_address']:
                unique_ips_current_month.add(row['ip_address'])

            # Calculate total tokens used per row
            totalTokenUsed = 0
            totalTokenCost =0.0
            for response in row['responses']:
                totalTokenUsed += response.get('token_used', 0)
                totalTokenCost += response.get('total_cost', 0.0)

            total_tokens_used += totalTokenUsed
            total_token_cost += totalTokenCost
             # Count unique IPs
            if ip_address:
                total_unique_ips += 1

            # Track monthly unique users
            if ip_address:
                if to_date_str:
                    month_to_date = to_date.strftime('%Y-%m')

                    if month_to_date == month_part:
                        unique_ips_current_month.add(ip_address)
                elif month_part == current_month:
                    unique_ips_current_month.add(ip_address)
            
            
            # Track unique sessions by IP
            ip_address = row.get('ip_address')
            session_id = row.get('session_id')
            if  session_id and ip_address:
                if session_id not in sessions_by_ip[ip_address]:
                    sessions_by_ip[ip_address].add(session_id)
                    total_sessions += 1
                unique_ips.add(ip_address)
                
                # Track first access date by IP
                if ip_address not in first_access_by_ip or dt < first_access_by_ip[ip_address]:
                    first_access_by_ip[ip_address] = dt
            
            
            # Calculate the number of unique IPs
        total_unique_ips = len(unique_ips)
        # Calculate average sessions per IP
        average_sessions_per_ip = total_sessions / total_unique_ips if total_unique_ips > 0 else 0
        
        # Print or return the result
        average_sessions_per_ip = round(average_sessions_per_ip, 2)
    
        # Calculate average tokens per IP
        average_tokens_per_ip = total_tokens_used / total_unique_ips if total_unique_ips else 0
        average_tokens_per_ip = round(average_tokens_per_ip,2)

        # Calculate average tokens per Session
        average_tokens_per_session = total_tokens_used / len(results)
        average_tokens_per_session = round(average_tokens_per_session,2)

        # Calculate average token cost per user
        average_token_cost_per_user = total_token_cost / total_unique_ips if total_unique_ips else 0
        average_token_cost_per_user =round(average_token_cost_per_user,4)

        for row in data:
            # Parse the string into a datetime object
            datetime_str = row['datetime']
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
            
            # Extract the date and time
            date_part = dt.date()
            time_part = dt.time()
            month_part = dt.strftime('%Y-%m')

            # Format the date and time
            formatted_date = date_part.strftime('%Y-%m-%d')
            formatted_time = time_part.strftime('%H:%M:%S')
            
            totalTokenUsed = 0
            totalTokenCost = 0.0

            # Iterate over the responses to sum the token_used and total_cost
            for response in row['responses']:
                totalTokenUsed += response['token_used']
                totalTokenCost += response['total_cost']

            totalTokenCost = round(totalTokenCost, 4)

            # Add session ID to the set of unique sessions
            unique_sessions.add(row['session_id']) 
            result.append({
                'date': formatted_date,
                'time': formatted_time,
                'session_id': row['session_id'],
                'ip_address': row['ip_address'],
                'total_token_used': totalTokenUsed,
                'total_token_cost': totalTokenCost
            })

        # Calculate frequency of use
        total_sessions = len(results)
        unique_ips_count = len(unique_ips_today) if from_date_str or to_date_str else len(set(row['ip_address'] for row in results if row['ip_address']))
        frequency_of_use = total_sessions / unique_ips_count if unique_ips_count else 0
        frequency_of_use = round(frequency_of_use, 2)
        
         # Calculate retention rate
        returning_ips = set()
        for row in sorted_results:
            datetime_str = row['datetime']
            dt = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
            ip_address = row['ip_address']

            if ip_address and ip_address in first_access_by_ip and dt > first_access_by_ip[ip_address]:
                returning_ips.add(ip_address)

        retention_rate = (len(returning_ips) / len(first_access_by_ip)) * 100 if first_access_by_ip else 0
        retention_rate = round(retention_rate, 2)

        # Prepare the response
        response_data = {
            'draw': draw,
            'recordsTotal': len(results),
            'recordsFiltered': paginator.count if hasattr(paginator, 'count') else 0,
            'data': list(result),
            'unique_sessions_count': len(unique_sessions),
            'daily_unique_ips': daily_unique_users_count,
            'monthly_unique_ips': len(distict_ips),
            'monthly_unique_ips_count': len(monthly_unique_ips),
            'unique_ips_today': len(unique_ips_today),
            'unique_ips_current_month': len(unique_ips_current_month),
            'frequency_of_use': frequency_of_use,
            'average_tokens_per_ip': average_tokens_per_ip,
            'average_token_cost_per_user': average_token_cost_per_user,
            'average_tokens_per_session': average_tokens_per_session,
            'average_sessions_per_ip':average_sessions_per_ip,
            'user_retention_rate': retention_rate
        }
        
        end_time = time.time()
        time_taken = end_time - start_time  # calculate time difference
        # print("Time taken to execute function: {:.4f} seconds".format(time_taken))

        return JsonResponse(response_data)
    except Exception as e:
        return JsonResponse({'error': str(e)})


@django.views.decorators.csrf.csrf_exempt   
def get_session_details(request):
    
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
    
def getUserData(request):
    print(f"Data: In getUserData function")

    try:
        users = User.objects.all()
        user_list = list(users.values())
        # print(f"Data: {user_list}")
    except Exception as e:
        print(f"Error fetching session details: {e}")
        user_list = []

    if user_list:
        return JsonResponse(user_list, safe=False)  # Assuming you want to return the items as a response
    else:
        return HttpResponse({})  # Return an empty dictionary if no items found    
    
@django.views.decorators.csrf.csrf_exempt   
def delete_user(request):
    if request.method == 'POST':
        user_id = request.POST.get('userId', None)
        
        if user_id:
            try:
                user = get_object_or_404(User, id=user_id)
                user.delete()
                return JsonResponse({'status': 200, 'message': 'User deleted successfully'})
            except User.DoesNotExist:
                return HttpResponseBadRequest('User does not exist')
            except Exception as e:
                return HttpResponseBadRequest(f'An error occurred: {str(e)}')
        else:
            return HttpResponseBadRequest('User ID not provided')
    else:
        return HttpResponseBadRequest('Invalid request method')


def logoutFuntion(request):
    logout(request)
    return redirect('login')  