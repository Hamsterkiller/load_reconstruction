from datetime import date
from model.launch_reconstruction_model import launch_reconstruction_model

if __name__ == '__main__':

    target_date = date(2023, 11, 1)
    connection_string = "mssql+pyodbc://calculator:calculator@195.133.69.14/SKMRUSMSSQL?driver=ODBC+Driver+17+for+SQL+Server"
    db_schema = 'model1'

    launch_reconstruction_model(target_date, 0, connection_string, db_schema)

