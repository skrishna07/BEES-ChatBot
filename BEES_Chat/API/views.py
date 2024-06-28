import datetime
import json
import traceback
import uuid
from azure.cosmos import CosmosClient, exceptions, PartitionKey
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
from .serializers import UserSerializer
from .SourceCode.BEES_QA import AzureCosmosQA
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

# Initialize Cosmos Chat History Container:
client = CosmosClient(os.getenv('WebChat_EndPoint'), os.getenv('WebChat_Key'))
database = client.get_database_client(os.getenv('WebChat_DB'))
History_container = database.get_container_client(os.getenv('WebChat_History_Container'))
History_container = database.create_container_if_not_exists(
    id=os.getenv('WebChat_History_Container'),
    partition_key=PartitionKey(path="/id"),
    offer_throughput=400
)


def update_chat_history(container, response, session_id, ip_address):
    # Try to find an existing record with the same session_id
    try:
        query_str = f"SELECT * FROM c WHERE c.session_id = '{session_id}'"
        items = list(container.query_items(query=query_str, enable_cross_partition_query=True))
    except:
        items = []

    if items:
        # Update the existing record
        existing_record = items[0]
        if 'responses' not in existing_record:
            existing_record['responses'] = []
        existing_record['responses'].append(response)
        print("exist response")
        # Replace the item in the container
        container.replace_item(item=existing_record['id'], body=existing_record)
    else:
        # Create a new record
        chat_history_item = {
            'id': str(session_id),
            'ip_address': ip_address,
            'session_id': session_id,
            'responses': [response],
            'datetime': str(datetime.datetime.now())
        }
        container.create_item(body=chat_history_item)


@api_view(['POST'])
def signup(request):
    serializer = UserSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        user = User.objects.get(username=request.data['username'])
        user.set_password(request.data['password'])
        user.save()
        token = Token.objects.create(user=user)
        return Response({'token': token.key, 'user': serializer.data})
    return Response(serializer.errors, status=status.HTTP_200_OK)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def regenerate_token(request):
    token = Token.objects.get(user=request.user)
    token.delete()
    new_token = Token.objects.create(user=request.user)
    return Response({'token': new_token.key})


@api_view(['POST'])
def login(request):
    user = get_object_or_404(User, username=request.data['username'])
    if not user.check_password(request.data['password']):
        return Response("missing user", status=status.HTTP_404_NOT_FOUND)
    token, created = Token.objects.get_or_create(user=user)
    serializer = UserSerializer(user)
    return Response({'token': token.key, 'user': serializer.data})


@api_view(['POST'])
@authentication_classes([SessionAuthentication, TokenAuthentication])
@permission_classes([IsAuthenticated])
def AIResponse(request):
    try:
        request_body = json.loads(request.body)
        query = request_body.get('query')
        session_id = request_body.get('session_id')
        ip_address = request_body.get('ip_address')
        if not session_id:
            request.session.create()
            session_id = request.session.session_key
        if not query:
            return Response({"error": "Query not provided"}, status=400)

        response_data, token_used, total_cost = AzureCosmosQA(query, session_id)
        response_dict_history = {'query': query, 'response': response_data, 'sourcelink': '',
                                 'status': 'success',
                                 'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        response_dict = {'response': response_data, 'sourcelink': '', 'session_id': session_id,
                         'status': 'success',
                         'token_used': token_used, 'total_cost': total_cost, 'statuscode': 200}
        # Save to Cosmos DB if response is successful
        update_chat_history(History_container, response_dict_history, session_id, ip_address)
        return Response(response_dict)

    except json.JSONDecodeError as e:
        return Response({"error": "Invalid JSON" + "\n e", 'status': 'Fail', 'statuscode': 400}, status=400)
    except exceptions.CosmosHttpResponseError as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)

    except Exception as e:
        return Response({"error": str(e), 'status': 'Fail', 'statuscode': 500}, status=500)
