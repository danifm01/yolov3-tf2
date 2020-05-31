import pandas as pd
from tqdm import tqdm
import os


def comp(value, minim, maxim):
    if minim <= value <= maxim:
        return True
    return False


if __name__ == '__main__':
    data = pd.read_csv('data\\airsim\\parametros.csv')
    data = data.set_index('nombre')
    # Ruta con todas las imágenes
    dirPath = os.path.join('data', 'airsim', 'SceneFull')
    image_list = os.listdir(dirPath)
    newdirPathTrain = os.path.join(dirPath, 'Train')
    newdirPathVal = os.path.join(dirPath, 'Validation')
    if not os.path.exists(newdirPathTrain):
        os.mkdir(newdirPathTrain)
    if not os.path.exists(newdirPathVal):
        os.mkdir(newdirPathVal)
    val = 0
    train = 0
    for ima in tqdm(image_list):
        if os.path.isfile(os.path.join(dirPath, ima)):
            dist = data.loc[ima.replace('.png', '')]['distancia']
            # Validation
            if comp(dist, 7.5, 12.5) or comp(dist, 17.5, 20.5) or comp(
                    dist, 27.5, 30.5):
                os.rename(os.path.join(dirPath, ima),
                          os.path.join(newdirPathVal, ima))
                val += 1
            # Training
            else:
                os.rename(os.path.join(dirPath, ima),
                          os.path.join(newdirPathTrain, ima))
                train += 1
    print(f'Train: {train}, validation: {val}, ratio: {train / (train + val)}')