#Modelo de pré-processamento otsu + adaptativa
import cv2
import numpy as np
import os


#Paths das imagens processadas e que serão salvas 
path_img_orig = r"D:/.../Dataset Clouds 1500/2021-12-01/img/11-31-00.jpg"
path_img_save = r"D:/.../ResultadoLimiarizacao"
nome_original = os.path.splitext(os.path.basename(path_img_orig))[0]

#Carregando a imagem e transformando em escala de cinza
img_orig = cv2.resize(cv2.imread(path_img_orig), (600, 600))
img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)  # CARREGA A IMAGEM, REDIMENSIONA E COLOCA NA ESCALA CINZA
img_blur = cv2.GaussianBlur(img, (5, 5), 0)

#Limiarização de Otsu
limiar, img_bin = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Limiarização adaptativa
blsize = 3
c = 2
img_adp = cv2.adaptiveThreshold(img_bin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blsize , c)

#Exibindo o limiar calculado no terminal
print(f"O Otsu definiu o limiar ideal em: {limiar}")

#Transformando as imagens para 3 canais, permitindo destaque nas legendas de configurações
img_bin_bgr = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
img_adp_bgr = cv2.cvtColor(img_adp, cv2.COLOR_GRAY2BGR)

#Configurações do texto das legendas
fonte = cv2.FONT_HERSHEY_SIMPLEX
escala = 0.8
cor_texto = (0, 0, 255)       # Vermelho em BGR
cor_fundo = (255, 255, 255)   # Branco para contorno das legendas
espessura = 2
posicao = (20, 40)            # Margem de 20px da esquerda e 40px do topo

#Inserindo os textos nas imagens
for imagem, nome in [
    (img_orig, "Original"),
    (img_bin_bgr, f"Otsu (Limiar: {int(limiar)})"),
    (img_adp_bgr, f"Adaptativa (BlSize={blsize}, C={c})")
]:
    # Texto de fundo (sombra/contorno)
    cv2.putText(imagem, nome, posicao, fonte, escala, cor_fundo, espessura + 2, cv2.LINE_AA)
    # Texto principal
    cv2.putText(imagem, nome, posicao, fonte, escala, cor_texto, espessura, cv2.LINE_AA)

# Resultado lado a lado (agora todas são BGR de 3 canais):
comparacao = np.hstack((img_orig, img_bin_bgr, img_adp_bgr))

# Código para salvar as imagens processadas:  
nome_saida = f"{nome_original}_Otsu{int(limiar)}_Mean_Block{blsize}_C{c}.png"
caminho_saida = os.path.join(path_img_save, nome_saida)
cv2.imwrite(caminho_saida, comparacao)
print(f"Imagem salva em:\n{caminho_saida}")

# Exibe a imagem final:
cv2.imshow("Comparacao:", comparacao)
cv2.waitKey(0)  # Para a janela ficar parada
cv2.destroyAllWindows()
