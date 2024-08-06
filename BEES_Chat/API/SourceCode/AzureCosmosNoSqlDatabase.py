from azure.cosmos import exceptions, CosmosClient, PartitionKey
from SourceCode.Log import Logger
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
logger = Logger()


class CosmosDBManager:
    def __init__(self, endpoint, master_key):
        try:
            self.client = CosmosClient(endpoint, master_key)
            logger.log("Successfully connected to Cosmos DB", "Info")
        except exceptions as e:
            error_details = logger.log("Error initializing Cosmos DB client:", "Error")
            raise Exception(error_details)

    def create_container(self, database_id, container_id, partition_key_path, offer_throughput=400):
        database = self.client.get_database_client(database_id)
        try:
            container = database.create_container(
                id=container_id,
                partition_key=PartitionKey(path=partition_key_path),
                offer_throughput=offer_throughput
            )
            logger.log(f"Container '{container_id}' created in database '{database_id}'.", "Info")
            return container
        except exceptions.CosmosResourceExistsError:
            logger.log(f"Container '{container_id}' already exists in database '{database_id}'.", "Info")
            container = database.get_container_client(container_id)
            return container
        except exceptions as e:
            error_details = logger.log(f"Error creating container: {e}", "Error")
            raise Exception(error_details)

    def insert_record(self, container, item):
        try:
            container.create_item(item)
            logger.log(f"Successfully inserted record in container", "Info")
        except exceptions as e:
            error_details = logger.log(f"Error inserting container: {e}", "Error")
            raise Exception(error_details)

    def update_record(self, container, item_id, updated_data):
        try:
            item = container.read_item(item_id=item_id)
            item.update(updated_data)
            container.replace_item(item=item, item_read_version=item['etag'])
            logger.log(f"Record '{item_id}' updated successfully.", "Info")
        except exceptions as e:
            error_details = logger.log(f"Error updating record: {e}", "Info")
            raise Exception(error_details)

