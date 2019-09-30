import sqlite3
def FetchData():
    connection =  sqlite3.connect('./Database/Products.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM BillData')
    row = cursor.fetchone()
    data = []
    while row is not None:
        data.append(row)
        row = cursor.fetchone()
    
    print(data)
    connection.commit()
    return data

