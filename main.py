import cv2

# Loading the trained face detector
# Carregando o detector de faces terinado
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Setting video input and display window size
# Configurando a entrada de vídeo e ajustando o tamanho da janela de exibição
video_input = cv2.VideoCapture(0)
video_input.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_input.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initializing the recognition tool and providing the trained model
# Inicializando a ferramenta de reconhecimento e fornecendo o modelo treinado
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
font = cv2.FONT_HERSHEY_SIMPLEX

names = ['THIAGO']
sampleId = 0

# Execution loop, will be stopped when user press "q" key
# Loop de execução, será interrompido quando o usuário pressionar a tecla "q"
while not cv2.waitKey(20) & 0xFF == ord('q'):
    
    # Capturing images from video input
    # Capturando as imagens a partir da entrada de vídeo
    ret, color_frame = video_input.read()
    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    # Applying face detector on the video input
    # Aplicando detector de faces sobre o video
    detected_faces = face_detector.detectMultiScale(gray_frame)
    
    # Contouring the detected face
    # Contornando a face detectada
    for x, y, w, h in detected_faces:
        cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        num, confidence = recognizer.predict(gray_frame[y:y+h, x:x+w])

        if (confidence < 100):
            id = names[num]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "UNKNOW | DESCONHECIDO"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(color_frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(color_frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) 
    
    # Displaying the count of detected faces
    # Exibindo a contagem de faces detectadas
    cv2.putText(img=color_frame,
                text='Detected Faces | Faces detectadas:' + str(len(detected_faces)),
                org=(40, 40),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0,255,0),
                thickness=2)      
    
    # Showing the video
    # Exibindo o video
    cv2.imshow('color', color_frame)


