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


def get_policy_details(SQLDatabase, head_id):
    try:
        rows = SQLDatabase.select_data('Policy', (
            "PolicyCode,PolicyName"), "id="+str(head_id))
        logger.log("Attachment policy data successfully fetched", "Info")
        if rows:
            return rows[0]
        else:
            return rows
    except Exception as e:
        logger.log(f"Error fetching attachment data: {str(e)}", "Error")
        raise


def get_attachments_data(SQLDatabase):
    try:
        if os.getenv('InsertAllData') == 'Y':
            query = f"""
                    SELECT fc.Id, fc.FileName, fc.FilePath, fc.IsActive, fc.ChangedOn, fc.HeadId, fc.FileCategoryId, fc.LinkURL
                    FROM [Attachment] AS fc
                    INNER JOIN [FileCategory] AS c ON fc.[FileCategoryId] = c.id
                    WHERE c.[Name] in {os.getenv('AttachmentCategory')};"""
        else:
            query = f"""
                    SELECT fc.Id, fc.FileName, fc.FilePath, fc.IsActive, fc.ChangedOn, fc.HeadId, fc.FileCategoryId, fc.LinkURL
                    FROM [Attachment] AS fc
                    INNER JOIN [FileCategory] AS c ON fc.[FileCategoryId] = c.id
                    WHERE c.[Name] in {os.getenv('AttachmentCategory')} AND fc.ChangedOn >= DATEADD(day, -7, GETDATE()) 
                    AND fc.ChangedOn <= GETDATE();"""
        rows = SQLDatabase.select_data_with_join(query)
        logger.log("Attachment data successfully fetched", "Info")
        return rows
    except Exception as e:
        logger.log(f"Error fetching attachment data: {str(e)}", "Error")
        raise


def get_category_name(SQLDatabase, File_id):
    try:
        rows = SQLDatabase.select_data('FileCategory', (
            "Name"), f'Id={File_id}')
        if rows:
            return rows[0]
        else:
            return rows
    except Exception as e:
        logger.log(f"Error fetching file category data: {str(e)}", "Error")
        raise


def process_attachments(SQLDatabase, container):
    try:
        # Server User Details:
        user_name = getpass.getuser()
        machine_name = socket.gethostname()

        attachments = get_attachments_data(SQLDatabase)

        for attachment in attachments:
            try:
                head_id = attachment['HeadId']
                file_category_id = attachment['FileCategoryId']
                if file_category_id == 13:
                    policydata = get_policy_details(SQLDatabase, head_id)
                id = attachment['Id']
                query = f"SELECT * FROM c WHERE c.id = 'Attachment_{id}-{file_category_id}-{head_id}'"
                try:
                    query_items = container.query_items(query, enable_cross_partition_query=True)
                    items = list(query_items)
                except:
                    items = []
                    logger.log(f"Attachment data not exists in master: {str(e)}", "Error")
                if not items:
                    if attachment['ChangedOn']:
                        ChangedOn = str(attachment['ChangedOn'].replace(microsecond=0))
                    else:
                        ChangedOn = None
                    Category = get_category_name(SQLDatabase, file_category_id)
                    if Category:
                        Category = Category['Name']
                    else:
                        Category = None
                    if attachment['IsActive']:
                        RunFlag = "Y"
                    else:
                        RunFlag = "N"
                    new_item = {
                        "id": f'Attachment_{id}-{file_category_id}-{head_id}',
                        "ID": f'Attachment_{id}-{file_category_id}-{head_id}',
                        "RunFlag": RunFlag,
                        "ISActive": str(attachment['IsActive']),
                        "Type": "Attachment",
                        "ChangedOn": ChangedOn,
                        "FilePath": str(attachment['FilePath']),
                        "LinkURL": str(attachment['LinkURL']),
                        "PageContent": "",
                        "PageTitle": "",
                        "URL": "",
                        "NewsContent": "",
                        "NewsTitle": "",
                        "CreatedBy": str(user_name) + "_" + str(machine_name),
                        "CreatedOn": datetime.now(),
                        "UpdatedBy": "",
                        "UpdatedOn": "",
                        "Category": Category,
                        "ExceptionDetails": "",
                        "policycode": policydata['PolicyCode'],
                        "policyname": policydata['PolicyName']
                    }
                    container.create_item(body=serialize_item(new_item))
                else:
                    existing_item = items[0]
                    if attachment['IsActive']:
                        if existing_item['ChangedOn']:
                            existing_changed_on = datetime.strptime(existing_item['ChangedOn'], '%Y-%m-%d %H:%M:%S')
                            new_changed_on = attachment['ChangedOn'].replace(microsecond=0)
                            if new_changed_on > existing_changed_on:
                                existing_item['RunFlag'] = 'Y'
                                existing_item['ISActive'] = 'True'
                                existing_item['ChangedOn'] = str(new_changed_on)
                                existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                                existing_item['UpdatedOn'] = datetime.now()
                                container.replace_item(existing_item['id'], serialize_item(existing_item))
                        else:
                            if attachment['ChangedOn']:
                                new_changed_on = str(attachment['ChangedOn'].replace(microsecond=0))
                                existing_item['RunFlag'] = 'Y'
                                existing_item['ISActive'] = 'True'
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
                logger.log(f"Error inserting attachments Attachment: {str(e)}", "Error")
    except Exception as e:
        logger.log(f"Error processing attachments: {str(e)}", "Error")
        raise


def Load_Data(SQLDatabase, container):
    try:
        process_attachments(SQLDatabase, container)
    except Exception as e:
        error_details = logger.log(f"Unhandled exception in load attachment data: {str(e)}", "Error")
        raise Exception(error_details)
