# <YOUR_IMPORTS>
import json
import os
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '..')
def predict():
    file_name = f'{path}/data/models/cars_pipe.pkl'
    with open(file_name, 'rb') as file:
        object_to_load = dill.load(file)
    #print(object_to_load)

    directory_path = f'{path}/data/test'  # Замените путь на вашу директорию

    file_list = os.listdir(directory_path)

    result = []

    for file_name in file_list:
        file_path = os.path.join(directory_path, file_name)

        if os.path.isfile(file_path):
            print(f"Найден файл: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
                #print(data)

            df = pd.DataFrame([data])
            #print(df)

            y = object_to_load.predict(df)
            #print(y)

            result.append({'ID': data['id'], 'Predict': y[0]}) 
    #print(result)

    otv = pd.DataFrame(result)
    #print(otv)

    otv.to_csv(f'{path}/data/predictions/result.csv', index=False)


if __name__ == '__main__':
    predict()
