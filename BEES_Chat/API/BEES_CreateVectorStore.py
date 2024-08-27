import time
from SourceCode.Log import Logger
from SourceCode import Download_AzureBlobFiles
from SourceCode import Extract_PDF
from SourceCode import Load_Attachment_Data
from SourceCode import Load_Page_Data
from SourceCode import Load_News_Data
from SourceCode import AzureCosmosVectorStoreContianer
from SourceCode.Convert_text_to_doc import NewsDoc, PageDoc
from SourceCode.BEES_DB import SQLDatabase
from SourceCode.AzureCosmosNoSqlDatabase import CosmosDBManager
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import os
import getpass
import socket

load_dotenv(find_dotenv())


def serialize_item(item):
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


class BEES_Main:
    def __init__(self):
        self.BEES_Database = SQLDatabase(Server=os.getenv('BEES_DBServer'),
                                         Database=os.getenv('BEES_DB'),
                                         Username=os.getenv('BEES_Username'),
                                         Password=os.getenv('BEES_Password'))

        self.BEES_Database.connect()
        self.Logger = Logger()
        self.Cosmos_Client = CosmosDBManager(endpoint=os.getenv('WebChat_EndPoint'),
                                             master_key=os.getenv('WebChat_Key'))

        self.Cosmos_Client.create_database_if_not_exists(item=os.getenv('WebChat_DB'))

        self.ChatMainContainer = self.Cosmos_Client.create_container(database_id=os.getenv('WebChat_DB'),
                                                                     container_id='BEES_ChatMain',
                                                                     partition_key_path="/ID")

    def GetProcessData(self):
        try:
            query = f"SELECT * FROM c WHERE c.RunFlag = 'Y' OFFSET 0 LIMIT 1"
            try:
                query_items = self.ChatMainContainer.query_items(query, enable_cross_partition_query=True)
                items = list(query_items)
                ProcessData = items[0]
                self.Logger.log(f"Record Fetched - {ProcessData['id']} ", "Info")
            except:
                ProcessData = None
            return ProcessData
        except Exception as e:
            error_details = self.Logger.log(f"error occurred in fetching process data: {str(e)}", "Error")
            raise Exception(error_details)

    def updateData(self, Data):
        try:
            user_name = getpass.getuser()
            machine_name = socket.gethostname()
            Data['RunFlag'] = 'N'
            Data['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
            Data['UpdatedOn'] = datetime.now()
            Data['ExceptionDetails'] = ''
            self.ChatMainContainer.replace_item(Data['id'], serialize_item(Data))
        except Exception as e:
            error_details = self.Logger.log(f"error occurred in updating process data: {str(e)}", "Error")
            raise Exception(error_details)

    def updateException(self, Data, details, RunFlag):
        try:
            user_name = getpass.getuser()
            machine_name = socket.gethostname()
            Data['RunFlag'] = RunFlag
            Data['UpdatedBy'] = str(user_name) + "_" + str(machine_name)
            Data['UpdatedOn'] = datetime.now()
            Data['ExceptionDetails'] = str(details)
            self.ChatMainContainer.replace_item(Data['id'], serialize_item(Data))
        except Exception as e:
            error_details = self.Logger.log(f"error occurred in updating exception process data: {str(e)}", "Error")
            raise Exception(error_details)

    def run(self):
        try:
            self.Logger.log(f"Vectorstore data sync process started", "Info")
            Load_Attachment_Data.Load_Data(self.BEES_Database, self.ChatMainContainer)
            Load_Page_Data.Load_Data(self.BEES_Database, self.ChatMainContainer)
            Load_News_Data.Load_Data(self.BEES_Database, self.ChatMainContainer)
            while True:
                # Fetch new process record
                Data = self.GetProcessData()
                try:
                    if Data:
                        # Delete existing item
                        AzureCosmosVectorStoreContianer.delete_chunk_item(Data['id'])

                        # Process Attachment
                        if Data["Type"] == "Attachment":
                            filepath = os.path.join(os.getenv('Filepath'), Data["FilePath"])
                            file_exist = os.path.exists(filepath)
                            if "InternalPages" in filepath or "News" in filepath:
                                self.updateData(Data=Data)
                                continue
                            print(filepath)
                            # filepath, file_exist = Download_AzureBlobFiles.Download_File(Data["FilePath"])
                            if file_exist:
                                Pdf_content = Extract_PDF.process_documents(filepath, Data["Category"], Data["id"],
                                                                            Data["policycode"], Data["policyname"])
                                if Pdf_content[0].page_content == '':
                                    error_details = f'Data is empty - {Data["id"]}'
                                    self.updateException(Data, error_details, 'B')
                                    self.Logger.log(error_details, "Info")
                                    continue
                                else:
                                    AzureCosmosVectorStoreContianer.Load_ChunkData(Pdf_content)
                                    if Data["Category"] == "Policy":
                                        data = []
                                        metadata = {'source': filepath, 'category': Data["Category"],
                                                    'unique_id': Data["id"]}
                                        data.append(
                                            Document(page_content=Data["policycode"] + "-" + Data["policyname"],
                                                     metadata=metadata))
                                        AzureCosmosVectorStoreContianer.Load_ChunkData(data)
                            else:
                                error_details = f'Attachment File not exists for - {Data["id"]}'
                                self.updateException(Data, error_details, 'B')
                                self.Logger.log(error_details, "Info")
                                continue

                        # Process PageDetails
                        elif Data["Type"] == "page":
                            Page_Content = PageDoc(Data["PageContent"], Data["PageTitle"], Data["id"], Data["Category"],
                                                   Data["FilePath"])
                            if Page_Content[0].page_content == '':
                                error_details = f'Data is empty - {Data["id"]}'
                                self.updateException(Data, error_details, 'B')
                                self.Logger.log(error_details, "Info")
                                continue
                            else:
                                AzureCosmosVectorStoreContianer.Load_ChunkData(Page_Content)

                        # Process News
                        elif Data["Type"] == "News":
                            News_Content = NewsDoc(Data["NewsContent"], Data["NewsTitle"], Data["id"], Data["Category"])
                            if News_Content[0].page_content == '':
                                error_details = f'Data is empty - {Data["id"]}'
                                self.updateException(Data, error_details, 'B')
                                self.Logger.log(error_details, "Info")
                                continue
                            else:
                                AzureCosmosVectorStoreContianer.Load_ChunkData(News_Content)

                        # Update Process Completion
                        self.updateData(Data=Data)
                        self.Logger.log(f'Record processed successfully - f{Data["id"]}', "Info")
                    else:
                        # Stop Process
                        self.Logger.log(f"No Record", "Info")
                        break

                except Exception as e:
                    error_details = self.Logger.log(f"error occurred in updating exception process data: {str(e)}",
                                                    "Error")
                    self.updateException(Data, error_details, 'E')

            self.Logger.log(f"Vectorstore data sync process completed", "Info")
        except Exception as e:
            self.Logger.log(f"Unexpected error occurred: {str(e)}", "Error")

        finally:
            if self.BEES_Database:
                self.BEES_Database.close_connection()


if __name__ == "__main__":
    Master = BEES_Main()
    start_time = time.time()
    Master.run()
    end_time = time.time()
    # Calculate the time difference
    elapsed_time = end_time - start_time
    print(f"Time elapsed between lines: {elapsed_time} seconds")
