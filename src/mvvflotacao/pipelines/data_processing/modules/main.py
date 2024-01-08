import pandas as pd
# import requests
# from sqlalchemy import create_engine


class DataSourceReader:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.source_type = kwargs.get('source_type')

        self.data = None

    def read_data(self):
        if self.source_type == 'cloud':
            pass
        elif self.source_type == 'local':
            file_path = self.kwargs.get('file_path')
            if 'xlsx' in self.kwargs.keys():
                xlsx = self.kwargs.get('xlsx')
                self.data = pd.read_excel(file_path, **xlsx)
            else:
                self.data = pd.read_excel(file_path)
        elif self.source_type == 'api_sql':
            pass
        else:
            pass

        return self.data
