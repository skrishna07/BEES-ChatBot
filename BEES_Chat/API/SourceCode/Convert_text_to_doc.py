import pandas as pd
from bs4 import BeautifulSoup
from SourceCode.Log import Logger

logger = Logger()


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


def NewsDoc(html_content, NewsTitle, unique_id, category, date, URL):
    try:
        data = []
        # Parse the HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        # Extract and print the text content
        text_content = soup.get_text()
        # file_path = f"News?newsId={unique_id[5:]}"
        text_content = text_content.replace('\\n', '')
        text_content = text_content.replace('@', '')
        metadata = {'source': URL, 'category': category, 'unique_id': unique_id, 'NewsTitle': NewsTitle,
                    'Date': date}
        data.append(Document(page_content=text_content, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error creating document for news content: {str(e)}", "Error")
        raise Exception(error_details)


def PageDoc(text_content, PageTitle, unique_id, category, URL, date):
    try:
        data = []
        # Parse the HTML content
        soup = BeautifulSoup(text_content, 'html.parser')
        URL = str(URL)
        # Extract and print the text content
        text_content = soup.get_text()
        text_content = text_content.replace('\\n', '')
        text_content = text_content.replace('@', '')
        metadata = {'source': URL, 'category': category, 'unique_id': unique_id, 'PageTitle': PageTitle, 'Date': date}
        data.append(Document(page_content=text_content, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error creating document for page content: {str(e)}", "Error")
        raise Exception(error_details)


def LinksDoc(text_content, unique_id, category, URL, date):
    try:
        data = []
        metadata = {'source': URL, 'category': category, 'unique_id': unique_id, 'Date': date}
        data.append(Document(page_content=text_content, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error creating document for link content: {str(e)}", "Error")
        raise Exception(error_details)


def BusRouteGroup(text_content):
    # Parse the HTML content
    soup = BeautifulSoup(text_content, 'html.parser')

    # Find the table
    table = soup.find('table')

    # Extract headers
    headers = [th.text for th in table.find_all('th')]

    # Extract rows
    data = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cols = [td.text for td in row.find_all('td')]
        data.append(cols)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Group by 'Bus Route No' and aggregate rows into lists
    grouped_df = df.groupby(['Bus Route No', 'Shift', 'From', 'To', 'Bus Route', 'Co Ordinator', 'Mobile', 'URL']).agg(
        lambda x: x.tolist()).reset_index()
    return grouped_df


def BusRouteDoc(grouped_df, row, unique_id, category, date):
    try:
        data = []
        text_data = ''
        url_value = ''
        for header, value in zip(grouped_df.columns, row):
            if header == 'URL':  # Assuming 'URL' is the column name for URLs
                url_value = value  # Store only URL value
            text_data += f"{header}: {value}\n"  # Append the current row's data to all_data
        metadata = {'source': url_value, 'category': category, 'unique_id': unique_id, 'Date': date}
        data.append(Document(page_content=text_data, metadata=metadata))
        return data
    except Exception as e:
        error_details = logger.log(f"Error creating document for Bus route content: {str(e)}", "Error")
        raise Exception(error_details)
