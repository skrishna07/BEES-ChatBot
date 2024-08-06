import os
import datetime
import sys
import traceback


class Logger:

    def __init__(self):
        self.log_file = ""
        self.log_directory = ""

    def create_log_file(self):
        try:
            if not os.path.exists(self.log_directory):
                os.makedirs(self.log_directory)
            if not os.path.exists(self.log_file):
                with open(self.log_file, 'w'):  # Create the file if it doesn't exist
                    pass
        except OSError as e:
            print(f"Error occurred while creating log file: {e}")

    def log(self, message, level="Info"):
        try:
            current_dir = os.getcwd()
            self.log_directory = os.path.join(current_dir, "Log")
            self.log_file = os.path.join(self.log_directory,
                                         f"BEES_log_{datetime.datetime.now().strftime('%Y%m%d')}.txt")
            self.create_log_file()
            log_message = f"[{level}] [{datetime.datetime.now()}] {message}\n"
            print(log_message)
            error_message = ''
            if level.upper() == "ERROR":
                # Log error details with file name and line number
                exc_type, exc_obj, exc_tb = sys.exc_info()
                file_name = exc_tb.tb_frame.f_code.co_filename
                line_number = exc_tb.tb_lineno
                error_details = traceback.format_exc().splitlines()[-1]
                error_message = f"[{level}] {file_name}:LineNo-{line_number},Details - {error_details}\n"
                print(error_message)
            with open(self.log_file, "a") as f:
                f.write(log_message)
                if level.upper() == "ERROR":
                    f.write(error_message)
            return error_message
        except Exception as e:
            print(f"Error occurred while logging: {e}")
