from getData import FetchData

def BillData():
    products = []
    data = FetchData()
    for d in data:
        products.append({
            'id': d[0],
            'name': d[1],
            'weight': d[2],
            'quantity': 1,
            'price': d[3]
        })

    return products