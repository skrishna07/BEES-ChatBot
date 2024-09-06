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


def get_QuickLink_content_data(SQLDatabase):
    
    try:
        if os.getenv('InsertAllData') == 'Y':
            rows = SQLDatabase.select_data_with_join("""SELECT QuickLinkName, LinkURL, IsActive, ChangedOn, Id
                    FROM (
                      SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY QuickLinkName ORDER BY id) AS row_num
                      FROM QuickLinkData WHERE LinkURL not in ('#','NULL','')
                    ) AS subquery
                    WHERE row_num = 1""")
        else:
            rows = SQLDatabase.select_data_with_join("""SELECT QuickLinkName, LinkURL, IsActive, ChangedOn, Id
                    FROM (
                      SELECT *,
                        ROW_NUMBER() OVER (PARTITION BY QuickLinkName ORDER BY id) AS row_num
                      FROM QuickLinkData WHERE ChangedOn >= DATEADD(day, -7, GETDATE()) AND ChangedOn <= GETDATE() AND LinkURL not in ('#','NULL','')
                    ) AS subquery
                    WHERE row_num = 1""")
        logger.log("QuickLink_content data successfully fetched", "Info")
        return rows
    except Exception as e:
        logger.log(f"Error fetching QuickLink_content data: {str(e)}", "Error")
        raise


def process_QuickLink(SQLDatabase, container):
    try:
        # Server User Details:
        user_name = getpass.getuser()
        machine_name = socket.gethostname()
        QuickLinks = get_QuickLink_content_data(SQLDatabase)
        for link in QuickLinks:
            try:
                query = f"SELECT * FROM c WHERE c.id = 'QuickLink_{link['Id']}'"
                try:
                    query_items = container.query_items(query, enable_cross_partition_query=True)
                    items = list(query_items)
                except:
                    items = []
                    print("empty item")

                if not items:
                    print("new item")
                    if link['ChangedOn']:
                        ChangedOn = str(link['ChangedOn'].replace(microsecond=0))
                    else:
                        ChangedOn = None
                    Category = "QuickLink"
                    if link['IsActive']:
                        RunFlag = "Y"
                        IsActive = "True"
                    else:
                        RunFlag = "N"
                        IsActive = "False"
                    new_item = {
                        "id": f"QuickLink_{link['Id']}",
                        "ID": f"QuickLink_{link['Id']}",
                        "RunFlag": RunFlag,
                        "ISActive": IsActive,
                        "Type": "QuickLink",
                        "ChangedOn": ChangedOn,
                        "FilePath": link['LinkURL'],
                        "LinkURL": "",
                        "PageContent": "",
                        "PageTitle": "",
                        "URL": "",
                        "NewsContent": link['QuickLinkName'],
                        "NewsTitle": link['QuickLinkName'],
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
                    if link['IsActive']:
                        if existing_item['ChangedOn']:
                            existing_changed_on = datetime.strptime(existing_item['ChangedOn'], '%Y-%m-%d %H:%M:%S')
                            new_changed_on = link['ChangedOn'].replace(microsecond=0)
                            if new_changed_on > existing_changed_on:
                                existing_item['RunFlag'] = 'Y'
                                existing_item['IsActive'] = 'True'
                                existing_item['ChangedOn'] = str(new_changed_on)
                                existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                                existing_item['UpdatedOn'] = datetime.now()
                                container.replace_item(existing_item['id'], serialize_item(existing_item))
                        else:
                            if link['ChangedOn']:
                                new_changed_on = str(link['ChangedOn'].replace(microsecond=0))
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
                logger.log(f"Error inserting  QuickLink: {str(e)}", "Error")
                raise
    except Exception as e:
        logger.log(f"Error processing QuickLink: {str(e)}", "Error")
        raise


def Load_Data(SQLDatabase, container):
    try:
        process_QuickLink(SQLDatabase, container)
    except Exception as e:
        error_details = logger.log(f"Unhandled exception in load quick link data: {str(e)}", "Error")
        raise Exception(error_details)
