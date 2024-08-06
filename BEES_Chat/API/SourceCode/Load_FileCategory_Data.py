import getpass
import socket
from datetime import datetime
from SourceCode.Log import Logger

logger = Logger()


def serialize_item(item):
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


def get_Category_data(SQLDatabase):
    try:
        rows = SQLDatabase.select_data('FileCategory', (
            "Name, ChangedOn, Id, IsActive"))
        logger.log("Category data successfully fetched", "Info")
        return rows
    except Exception as e:
        logger.log(f"Error fetching Category data: {str(e)}", "Error")
        raise


def process_category(SQLDatabase, container):
    try:
        # Server User Details:
        user_name = getpass.getuser()
        machine_name = socket.gethostname()

        categories = get_Category_data(SQLDatabase)
        for category in categories:
            try:
                query = f"SELECT * FROM c WHERE c.id = '{category['Id']}'"
                try:
                    query_items = container.query_items(query, enable_cross_partition_query=True)
                    items = list(query_items)
                except:
                    items = []
                if not items:
                    if category['ChangedOn']:
                        ChangedOn = str(category['ChangedOn'].replace(microsecond=0))
                    else:
                        ChangedOn = None
                    new_item = {
                        "id": str(category['Id']),
                        "ISActive": category['IsActive'],
                        "ChangedOn": ChangedOn,
                        "Name": category['Name'],
                        "CreatedBy": str(user_name) + "_" + str(machine_name),
                        "CreatedOn": datetime.now(),
                        "UpdatedBy": "",
                        "UpdatedOn": ""
                    }
                    container.create_item(body=serialize_item(new_item))
                else:
                    # Record exists, check if it needs to be updated
                    existing_item = items[0]
                    if existing_item['ChangedOn']:
                        existing_changed_on = datetime.strptime(existing_item['ChangedOn'], '%Y-%m-%d %H:%M:%S')
                        new_changed_on = category['ChangedOn'].replace(microsecond=0)
                        if new_changed_on > existing_changed_on:
                            existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                            existing_item['UpdatedOn'] = datetime.now()
                            container.replace_item(existing_item['id'], serialize_item(existing_item))
                    else:
                        if category['ChangedOn']:
                            new_changed_on = str(category['ChangedOn'].replace(microsecond=0))
                            existing_item['RunFlag'] = 'Y'
                            existing_item['ChangedOn'] = new_changed_on
                            existing_item['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
                            existing_item['UpdatedOn'] = datetime.now()
                            container.replace_item(existing_item['id'], serialize_item(existing_item))
            except Exception as e:
                logger.log(f"Error inserting category: {str(e)}", "Error")
                raise
    except Exception as e:
        logger.log(f"Error processing pages: {str(e)}", "Error")
        raise


def Load_Data(SQLDatabase, container):
    try:
        process_category(SQLDatabase, container)
    except Exception as e:
        logger.log(f"Unhandled exception in load_filecategory data: {str(e)}", "Error")
