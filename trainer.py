import cv2
import numpy as np
from PIL import Image
import os
import sys

# Getting the data directory for training from the command line
# Recebendo o diretório de dados para treino da linha de comando
dataset = str(sys.argv[1])

# Loading the detection and classification model
# Carregando o modelo de detecção e classificação
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")


def getImagesForTraining(dataset):    

    """
    In this function, each sample image will be converted to grayscale, then
    in an array of NumPy. After that, an identifier is created for each sample from
    the samples directory name and, finally, face detection is performed on the sample.
    At the end of the block, samples and their identifiers are appended to lists
    samples and sampleId, and the function returns those lists.

    Nesta função, cada imagem de amostra será convertida para escala de cinza, em seguida
    em um array de NumPy. Após isso é criado um identificador para cada amostra a partir
    do nome do diretório das amostras, por fim, é realizada a detecção de face sobre a amostra.
    Na parte final do bloco, as amostras e seus identificadores são apendados nas listas
    samples e sampleId, e a função retorna essas listas.

    """

    samples = []
    sampleId = []
    samplePath = [os.path.join(dataset,i) for i in os.listdir(dataset)]

    for samplePath in samplePath:

        PIL_img = Image.open(samplePath).convert('L')
        sample_numpy = np.array(PIL_img,'uint8')

        faces = detector.detectMultiScale(sample_numpy)
        ids = int(str(dataset))
        for (x,y,w,h) in faces:
            samples.append(sample_numpy[y:y+h,x:x+w])
            sampleId.append(ids)

    return samples,sampleId

# Perform the training with provided data
# Realizadno o treinamento com os dados fornecidos
print ("\n Conducting training with the database | Realizando o treinamento com a base de dados ...")
samples, sampleId = getImagesForTraining(dataset)
recognizer.train(samples, np.array(sampleId))

# Saving the trained model
# Salvando o modelo teinado
print("\n Saving the trained model | Salvando o modelo treinado ...")
recognizer.write('trainer/trainer.yml')

# Displaying the number of faces detected in training
# Exibindo a quantidade de faces detectadas no treinamento
print("\n {0} face(s) trained | {0} face(s) treinada(s).".format(len(np.unique(sampleId))))
print("\n Training complete | Treinamento concluído")
