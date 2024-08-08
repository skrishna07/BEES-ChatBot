import pyodbc
from SourceCode.Log import Logger


class SQLDatabase:
    def __init__(self, Server, Database, Username, Password):
        self.logger = Logger()
        self.server = Server
        self.database = Database
        self.username = Username
        self.password = Password
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            if self.username and self.password:
                self.connection = pyodbc.connect(
                    f"DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password}")
            else:
                self.connection = pyodbc.connect(
                    f"DRIVER={{SQL Server}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
                )
            self.cursor = self.connection.cursor()
            self.logger.log(f"Database connected successfully", "Info")
        except pyodbc.Error as e:
            error_details = self.logger.log(f"Error DB connection: {str(e)}", "Error")
            raise Exception(error_details)

    def close_connection(self):
        try:
            if self.connection:
                self.connection.close()
                self.logger.log(f"Database disconnected successfully", "Info")
        except Exception as e:
            error_details = self.logger.log(f"Error DB close connection: {str(e)}", "Error")
            raise Exception(error_details)

    def restore_database(self):
        try:
            pass
        except Exception as e:
            error_details = self.logger.log(f"Error restore_database: {str(e)}", "Error")
            raise Exception(error_details)

    def insert_data(self, table, data):
        try:
            columns = ', '.join(data.keys())
            placeholders = ', '.join('?' * len(data))
            query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            query = query, list(data.values())
            # Unpack the tuple
            query, values = query
            self.cursor.execute(query, values)
            self.connection.commit()
        except pyodbc.Error as e:
            self.connection.rollback()
            error_details = self.logger.log(f"Error inserting data: {str(e)}", "Error")
            raise Exception(error_details)

    def update_data(self, table, data, Condition):
        try:
            set_values = ', '.join([f"{key} = ?" for key in data.keys()])
            query = f"UPDATE {table} SET {set_values} WHERE {Condition}"
            self.cursor.execute(query, list(data.values()))
            self.connection.commit()
        except pyodbc.Error as e:
            self.connection.rollback()
            error_details = self.logger.log(f"Error updating data: {str(e)}", "Error")
            raise Exception(error_details)

    def select_data(self, table, columns="*", condition=""):
        try:
            query = f"SELECT {columns} FROM {table}"
            if condition:
                query += f" WHERE {condition}"
            self.cursor.execute(query)
            columns = [column[0] for column in self.cursor.description]
            rows = self.cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            return data
        except pyodbc.Error as e:
            error_details = self.logger.log(f"Error selecting data: {str(e)}", "Error")
            raise Exception(error_details)

    def select_data_with_join(self, query):
        try:
            self.cursor.execute(query)
            columns = [column[0] for column in self.cursor.description]
            rows = self.cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
            return data
        except pyodbc.Error as e:
            error_details = self.logger.log(f"Error selecting data: {str(e)}", "Error")
            raise Exception(error_details)
