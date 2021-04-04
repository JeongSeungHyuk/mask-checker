import csv
import os


raw_dir = '/opt/ml/input/data/train'
target_dir = '/opt/ml/mask-checker/data'


with open(f'{raw_dir}/train.csv') as f:
    reader = csv.reader(f)

    dataset = []
    dataset.append(['gender', 'age', 'mask', 'image_path'])
    for row in list(reader)[1:]:
        gender = row[1]
        age = row[3]
        path = row[4]

        images = os.listdir(f'{raw_dir}/images/{path}')
        for image in images:
            if image[0] != '.':
                image_path = f'{path}/{image}'
                mask = image.split('.')[0]
                data = [gender, age, mask, image_path]
                dataset.append(data)

with open(f'{target_dir}/metadata.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(dataset)
