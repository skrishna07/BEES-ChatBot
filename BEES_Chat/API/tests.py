import re

path = "C:\\Users\\BRADSOL\\Documents\\Genai_ChatBot\\BEES_ChatV0.1\\Files\\BusRoute/169/BE - Bus Routes (2022) - General Shift - Route 34 (Time).jpg"
modified_path = re.sub(r'.*Files', '', path)
print(modified_path)
