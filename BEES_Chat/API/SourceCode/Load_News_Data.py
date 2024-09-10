import getpass
import socket
from datetime import datetime
from SourceCode.Log import Logger
from SourceCode.AzureCosmosVectorStoreContianer import delete_chunk_item
from dotenv import load_dotenv, find_dotenv
import os

logger = Logger()
load_dotenv(find_dotenv())


def serialize_item(item):
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


def get_News_content_data(SQLDatabase):
    try:
        if os.getenv('InsertAllData') == 'Y':
            rows = SQLDatabase.select_data('News', (
                "NewsTitle, NewsContent, IsActive, ChangedOn, Id, LinkUrl"))
        else:
            rows = SQLDatabase.select_data('News', (
                "NewsTitle, NewsContent, IsActive, ChangedOn, Id, LinkUrl"),
                                           "ChangedOn >= DATEADD(day, -7, GETDATE()) AND ChangedOn <= GETDATE()")
        logger.log("News_content data successfully fetched", "Info")
        return rows
    except Exception as e:
        logger.log(f"Error fetching News_content data: {str(e)}", "Error")
        raise


def process_News(SQLDatabase, container):
    try:
        # Server User Details:
        user_name = getpass.getuser()
        machine_name = socket.gethostname()
        News = get_News_content_data(SQLDatabase)
        for News_content in News:
            try:
                query = f"SELECT * FROM c WHERE c.id = 'News_{News_content['Id']}'"
                try:
                    query_items = container.query_items(query, enable_cross_partition_query=True)
                    items = list(query_items)
                except:
                    items = []
                if not items:
                    if News_content['ChangedOn']:
                        ChangedOn = str(News_content['ChangedOn'].replace(microsecond=0))
                    else:
                        ChangedOn = None
                    Category = "News"
                    if News_content['IsActive']:
                        RunFlag = "Y"
                        IsActive = "True"
                    else:
                        RunFlag = "N"
                        IsActive = "False"
                    new_item = {
                        "id": f"News_{News_content['Id']}",
                        "ID": f"News_{News_content['Id']}",
                        "RunFlag": RunFlag,
                        "ISActive": IsActive,
                        "Type": "News",
                        "ChangedOn": ChangedOn,
                        "FilePath": News_content['LinkUrl'],
                        "LinkURL": "",
                        "PageContent": "",
                        "PageTitle": "",
                        "URL": News_content['LinkUrl'],
                        "NewsContent": News_content['NewsContent'],
                        "NewsTitle": News_content['NewsTitle'],
                        "CreatedBy": str(user_name) + "_" + str(machine_name),
                        "CreatedOn": datetime.now(),
                        "UpdatedBy": "",
                        "UpdatedOn": "",
                        "Category": Category,
                        "ExceptionDetails": ""
                    }
                    container.create_item(body=serialize_item(new_item))
                else:
                    existing_item = items[0]
                    if News_content['IsActive']:
                        if existing_item['ChangedOn']:
                            existing_changed_on = datetime.strptime(existing_item['ChangedOn'], '%Y-%m-%d %H:%M:%S')
                            new_changed_on = News_content['ChangedOn'].replace(microsecond=0)
                            if new_changed_on > existing_changed_on:
                                existing_item['RunFlag'] = 'Y'
                                existing_item['IsActive'] = 'True'
                                existing_item['ChangedOn'] = str(new_changed_on)
                                existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                                existing_item['UpdatedOn'] = datetime.now()
                                container.replace_item(existing_item['id'], serialize_item(existing_item))
                        else:
                            if News_content['ChangedOn']:
                                new_changed_on = str(News_content['ChangedOn'].replace(microsecond=0))
                                existing_item['RunFlag'] = 'Y'
                                existing_item['IsActive'] = 'True'
                                existing_item['ChangedOn'] = new_changed_on
                                existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                                existing_item['UpdatedOn'] = datetime.now()
                                container.replace_item(existing_item['id'], serialize_item(existing_item))

                        # update error flag:
                        if existing_item['RunFlag'] == 'E':
                            existing_item['RunFlag'] = 'Y'
                            existing_item['ISActive'] = 'True'
                            existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                            existing_item['UpdatedOn'] = datetime.now()
                            container.replace_item(existing_item['id'], serialize_item(existing_item))
                    else:
                        # check data exists in chunk and delete Chunks Data
                        delete_chunk_item(existing_item['id'])
                        existing_item['RunFlag'] = 'N'
                        existing_item['ISActive'] = 'False'
                        existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                        existing_item['UpdatedOn'] = datetime.now()
                        container.replace_item(existing_item['id'], serialize_item(existing_item))
            except Exception as e:
                logger.log(f"Error inserting  news: {str(e)}", "Error")
                raise
    except Exception as e:
        logger.log(f"Error processing News: {str(e)}", "Error")
        raise


def Load_Data(SQLDatabase, container):
    try:
        process_News(SQLDatabase, container)
    except Exception as e:
        error_details = logger.log(f"Unhandled exception in load news data: {str(e)}", "Error")
        raise Exception(error_details)
