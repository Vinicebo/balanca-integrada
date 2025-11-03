import cv2
from ultralytics import YOLO
import serial
import time
import os

# --- 1. CONFIGURAÇÕES ---

# Caminho para o seu modelo treinado (baixe o 'best.pt' do seu Drive)
MODEL_PATH = "C:/Users/Pedro dos Santos/Documents/GitHub/balanca-integrada/best.pt"

# Descubra esta porta na sua Arduino IDE (em Ferramentas > Porta)
# Pode ser 'COM3' no Windows, ou '/dev/tty.usbmodem...' no Mac/Linux
PORTA_ARDUINO = "COM3"

# Dicionário de preços (exemplo baseado nas 60 classes do D2S)
# Você precisará preencher isso!
PRECOS_POR_KG = {
    "banana": 5.99,
    "apple": 8.99,
    "orange": 10.0,
    "gepa_bio_und_fair_fencheltee": 20.00, # Exemplo do seu dataset D2S
    # ... adicione as outras 56 classes ...
}

# --- 2. INICIALIZAÇÃO ---

# Carrega o modelo YOLOv8
if not os.path.exists(MODEL_PATH):
    print(f"ERRO: Arquivo do modelo não encontrado em {MODEL_PATH}")
    exit()

print("Carregando modelo de IA...")
model = YOLO(MODEL_PATH)
print("Modelo carregado com sucesso.")

# Conecta ao Arduino
try:
    arduino = serial.Serial(port=PORTA_ARDUINO, baudrate=9600, timeout=0.1)
    print(f"Conectado ao Arduino na porta {PORTA_ARDUINO}")
    time.sleep(2) # Espera a conexão estabilizar
except serial.SerialException as e:
    print(f"ERRO ao conectar ao Arduino: {e}")
    print("Verifique a porta serial e se o Arduino está conectado.")
    arduino = None

# Conecta à Câmera (0 é geralmente a webcam padrão)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível abrir a câmera.")
    exit()

# --- 3. LOOP PRINCIPAL DA APLICAÇÃO ---

print("\n--- Balança Inteligente Iniciada ---")
print("Pressione 'q' na janela da câmera para sair.")

peso_atual = 0.0

while True:
    # --- LEITURA DO ARDUINO ---
    if arduino and arduino.in_waiting > 0:
        try:
            # 1. Lê a linha (ex: b'0.152\r\n')
            linha_serial = arduino.readline()
            # 2. Decodifica para texto (ex: '0.152')
            peso_str = linha_serial.decode('utf-8').rstrip()
            # 3. Converte para número e armazena
            peso_atual = float(peso_str)
        except (ValueError, UnicodeDecodeError):
            pass # Ignora linhas mal formatadas

    # --- LEITURA DA CÂMERA ---
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da câmera.")
        break

    # --- PROCESSAMENTO DA IA ---
    # Roda a detecção. 'verbose=False' desliga os logs de print
    results = model(frame, conf=0.7, verbose=False)

    nome_item = "Nenhum item"
    preco_total = 0.0

    # Pega a detecção com maior confiança
    if results[0].boxes:
        best_det = results[0].boxes[0] # Pega a detecção principal
        class_id = int(best_det.cls)
        confianca = float(best_det.conf)
        nome_item = model.names[class_id]

        # Calcula o preço
        preco_unitario = PRECOS_POR_KG.get(nome_item, 0.0)
        preco_total = peso_atual * preco_unitario

    # --- EXIBIÇÃO NA TELA (GUI) ---
    # Desenha as caixas de detecção no frame
    frame_anotado = results[0].plot()

    # Cria um painel de informações
    # (cv2.putText(frame, "Texto", (x, y), fonte, tamanho, (cor_bgr), espessura))
    cv2.putText(frame_anotado, f"Item: {nome_item}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_anotado, f"Peso: {peso_atual:.3f} kg", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_anotado, f"Preco: R$ {preco_total:.2f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostra a janela
    cv2.imshow("Balança Inteligente - IA (Pressione 'q' para sair)", frame_anotado)

    # Condição de saída
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. FINALIZAÇÃO ---
cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()
print("Aplicação finalizada.")