# Face recognition system | Sistema de reconhecimento facial

This small face recognition system is capable of being trained to recognize up to 1 face, and can be used
for authentication purposes, for example.
Este pequeno sistema de reconhecimento facial é capaz de ser treinado para o reconhecimento de até 1 face,
podendo ser utilizado para fins de autenticação, por exemplo. 

## Training the system | Treinando o sistema
    
First, you must gather images of the person to be detected by the model in a directory located in the same
file directory "trainer.py". It is important that the directory is named "0".
Primeiro, deve-se reunir imagens da pessoa a ser detectada pelo modelo em um diretório localizado no mesmo
diretório do arquivo "trainer.py". É importante que o nome do diretório seja "0".

    
Then, run the "trainer.py" file, passing the directory with the images as an argument.
Em seguida, deve-se executar o arquivo "trainer.py", passando como argumento o diretório com as imagens.

After training, we can perform detections by running the "main.py" file. It is necessary to enter the name
that you want to give to the face detected in the "names" list.
Após o treinamento, podemos realizar detecções executando o arquivo "main.py". É necessário inserir o nome
que se deseja dar à face detectada na lista "names".
