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


def get_category_name(SQLDatabase, MenuID, MenuSubID, MenuChildID):
    try:
        if MenuChildID != 0:
            table = "MenuSubChild"
            File_id = MenuChildID
            column = "ChildName"
        elif MenuSubID != 0:
            table = "MenuSubCategory"
            File_id = MenuSubID
            column = "SubCategoryName"
        else:
            table = "MenuCategory"
            File_id = MenuID
            column = "CategoryName"
        rows = SQLDatabase.select_data(table, column, f'Id={File_id}')
        if rows:
            return rows[0][column]
        else:
            return rows
    except Exception as e:
        logger.log(f"Error fetching menu category data: {str(e)}", "Error")
        raise


def get_Page_data(SQLDatabase):
    try:
        if os.getenv('InsertAllData') == 'Y':
            rows = SQLDatabase.select_data('InternalPage', (
                "PageTitle, PageContent, IsActive, ChangedOn, MenuCategoryId, MenuSubCategoryId, MenuSubChildId, Id, URL"))
        else:
            rows = SQLDatabase.select_data('InternalPage', (
                "PageTitle, PageContent, IsActive, ChangedOn, MenuCategoryId, MenuSubCategoryId, MenuSubChildId, Id, URL"),
                                           "ChangedOn >= DATEADD(day, -7, GETDATE()) AND ChangedOn <= GETDATE()")
        logger.log("Page data successfully fetched", "Info")
        return rows
    except Exception as e:
        logger.log(f"Error fetching page data: {str(e)}", "Error")
        raise


def process_pages(SQLDatabase, container):
    try:
        # Server User Details:
        user_name = getpass.getuser()
        machine_name = socket.gethostname()

        pages = get_Page_data(SQLDatabase)
        for page in pages:
            try:
                query = f"SELECT * FROM c WHERE c.id = 'Page_{page['Id']}'"
                try:
                    query_items = container.query_items(query, enable_cross_partition_query=True)
                    items = list(query_items)
                except:
                    items = []
                if not items:
                    if page['ChangedOn']:
                        ChangedOn = str(page['ChangedOn'].replace(microsecond=0))
                    else:
                        ChangedOn = None
                    if page['IsActive']:
                        RunFlag = "Y"
                        IsActive = "True"
                    else:
                        RunFlag = "N"
                        IsActive = "False"
                    if int(page['MenuCategoryId']) == 0:
                        RunFlag = "N"
                        Category = "PageMenu"
                    else:
                        Category = get_category_name(SQLDatabase, int(page['MenuCategoryId']),
                                                     int(page['MenuSubCategoryId']), int(page['MenuSubChildId']))
                    new_item = {
                        "id": f"Page_{page['Id']}",
                        "ID": f"Page_{page['Id']}",
                        "RunFlag": RunFlag,
                        "ISActive": IsActive,
                        "Type": "page",
                        "ChangedOn": ChangedOn,
                        "FilePath": str(page["URL"]),
                        "LinkURL": "",
                        "PageContent": page['PageContent'],
                        "PageTitle": page['PageTitle'],
                        "URL": "",
                        "NewsContent": "",
                        "NewsTitle": "",
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
                    if page['IsActive']:
                        if existing_item['ChangedOn']:
                            existing_changed_on = datetime.strptime(existing_item['ChangedOn'], '%Y-%m-%d %H:%M:%S')
                            new_changed_on = page['ChangedOn'].replace(microsecond=0)
                            if new_changed_on > existing_changed_on:
                                if int(page['MenuCategoryId']) == 0:
                                    existing_item['RunFlag'] = 'N'
                                else:
                                    existing_item['RunFlag'] = 'Y'
                                existing_item['IsActive'] = 'True'
                                existing_item['ChangedOn'] = str(new_changed_on)
                                existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                                existing_item['UpdatedOn'] = datetime.now()
                                container.replace_item(existing_item['id'], serialize_item(existing_item))
                        else:
                            if page['ChangedOn']:
                                new_changed_on = str(page['ChangedOn'].replace(microsecond=0))
                                if int(page['MenuCategoryId']) == 0:
                                    existing_item['RunFlag'] = 'N'
                                else:
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
                logger.log(f"Error inserting  pages: {str(e)}", "Error")
    except Exception as e:
        logger.log(f"Error processing pages: {str(e)}", "Error")
        raise


def Load_Data(SQLDatabase, container):
    try:
        process_pages(SQLDatabase, container)
    except Exception as e:
        error_details = logger.log(f"Unhandled exception in load page data: {str(e)}", "Error")
        raise Exception(error_details)
