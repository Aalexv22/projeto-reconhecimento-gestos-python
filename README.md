# Projeto de Reconhecimento Facial e Gestos em Python

Este é um mini-projeto desenvolvido para explorar de forma descontraída o uso de **Python** e da biblioteca **MediaPipe** em aplicações de visão computacional.

## Objetivo
- Aprender mais sobre Python e bibliotecas de visão computacional.
- Detectar gestos com as mãos e expressões faciais.
- Transformar gestos em pequenas interações visuais (ex.: mostrar imagens diferentes para cada gesto).

## Tecnologias utilizadas
- Python 3.x
- OpenCV
- MediaPipe

## Funcionalidades
- Reconhecimento de gestos com as mãos (ex.: "L", mão aberta, indicador levantado).
- Reconhecimento facial básico (posição da cabeça).
- Exibição de imagens diferentes conforme o gesto detectado.

## Como executar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seuusuario/projeto-reconhecimento-gestos-python.git

pip install opencv-python mediapipe
python main.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== CONFIGURAÇÕES =====
modelo_maos = "hand_landmarker.task"
modelo_face = "face_landmarker.task"

imagem_L = cv2.imread("imagem.png")
imagem_duas_maos = cv2.imread("absolute_cinema.png")
imagem_patrick = cv2.imread("patrick.png")

# Detector de mãos
base_maos = python.BaseOptions(model_asset_path=modelo_maos)
opcoes_maos = vision.HandLandmarkerOptions(base_options=base_maos, num_hands=2)
detector_maos = vision.HandLandmarker.create_from_options(opcoes_maos)

# Detector de rosto
base_face = python.BaseOptions(model_asset_path=modelo_face)
opcoes_face = vision.FaceLandmarkerOptions(base_options=base_face,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False)
detector_face = vision.FaceLandmarker.create_from_options(opcoes_face)

camera = cv2.VideoCapture(0)

def dedos_abertos(mao):
    # Verifica se todos os dedos estão levantados
    dedos = [False]*5
    dedos[0] = mao[4].x > mao[3].x  # polegar
    dedos[1] = mao[8].y < mao[6].y  # indicador
    dedos[2] = mao[12].y < mao[10].y  # médio
    dedos[3] = mao[16].y < mao[14].y  # anelar
    dedos[4] = mao[20].y < mao[18].y  # mindinho
    return all(dedos)

while camera.isOpened():
    sucesso, frame = camera.read()
    if not sucesso:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    resultado_maos = detector_maos.detect(imagem_mp)
    resultado_face = detector_face.detect(imagem_mp)

    mostrar_L = False
    mostrar_duas_maos = False
    mostrar_patrick = False

    if resultado_maos.hand_landmarks:
        altura, largura, _ = frame.shape
        quantidade_maos = len(resultado_maos.hand_landmarks)

        # Duas mãos levantadas
        if quantidade_maos >= 2:
            mostrar_duas_maos = True

        # Gesto L
        for mao in resultado_maos.hand_landmarks:
            x_polegar = int(mao[4].x * largura)
            y_polegar = int(mao[4].y * altura)
            x_indicador = int(mao[8].x * largura)
            y_indicador = int(mao[8].y * altura)

            if abs(y_polegar - y_indicador) > 100 and abs(x_polegar - x_indicador) > 100:
                mostrar_L = True

            # Mão aberta + cabeça abaixada
            if dedos_abertos(mao) and resultado_face.face_landmarks:
                nariz = resultado_face.face_landmarks[0][1]  # landmark do nariz
                if nariz.y > 0.6:  # valor simples para "cabeça abaixada"
                    mostrar_patrick = True

            # desenha pontos da mão
            for ponto in mao:
                x = int(ponto.x * largura)
                y = int(ponto.y * altura)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # ===== MOSTRAR IMAGENS =====
    if mostrar_L and imagem_L is not None:
        img_redim = cv2.resize(imagem_L, (200, 200))
        frame[0:200, 0:200] = img_redim

    if mostrar_duas_maos and imagem_duas_maos is not None:
        img_redim = cv2.resize(imagem_duas_maos, (200, 200))
        frame[0:200, 220:420] = img_redim

    if mostrar_patrick and imagem_patrick is not None:
        img_redim = cv2.resize(imagem_patrick, (200, 200))
        frame[220:420, 0:200] = img_redim

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

camera.release()
cv2.destroyAllWindows()
