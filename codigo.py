import cv2
from ultralytics import YOLO
import serial
import time
import os

# --- 1. CONFIGURAÇÕES ---

# Caminho para o seu modelo treinado
MODEL_PATH = "C:/Users/vinic/OneDrive/Documentos/GitHub/balanca-integrada/best.pt"

# Porta serial do seu Arduino
PORTA_ARDUINO = "COM6"

# Dicionário de preços (exemplo)
PRECOS_POR_KG = {
    "banana": 5.99,
    "apple": 8.99,
    "orange": 2.99,
    "gepa_bio_und_fair_fencheltee": 20.00, # Exemplo do seu dataset D2S
    # ... adicione as outras 56 classes ...
}

# --- MUDANÇA (Novas Constantes) ---
# Quantos SEGUNDOS consecutivos a IA precisa acertar
# para que a detecção seja considerada "fixa"?
STABILITY_DURATION_SEC = 0.8 # Em segundos (ex: 0.8 segundos de detecção contínua)
CONFIDENCE_THRESHOLD = 0.6   # Limiar de confiança que você já estava usando
# --- FIM DA MUDANÇA ---


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

# --- MUDANÇA (Novas Variáveis de Estado) ---
# Variáveis para rastrear a estabilidade da detecção
peso_atual = 0.0
item_estavel = "Nenhum item"       # O que a GUI vai mostrar
confianca_estavel = 0.0
preco_total_estavel = 0.0

item_anterior = "Nenhum item"     # O que foi detectado no frame anterior
detection_start_time = None       # O carimbo de data/hora de quando a detecção começou
# --- FIM DA MUDANÇA ---


# --- 3. FUNÇÕES AUXILIARES ---

def ler_peso_da_balanca():
    """Lê uma linha da porta serial e retorna o peso como float."""
    if arduino and arduino.in_waiting > 0:
        try:
            linha_serial = arduino.readline()
            peso_str = linha_serial.decode('utf-8').rstrip()
            return float(peso_str) # Espera um float (ex: 0.152)
        except (ValueError, UnicodeDecodeError):
            return None # Ignora linhas mal formatadas
    return None

def desenhar_info_na_tela(frame, item, peso, preco, conf_str):
    """Desenha o painel de informações no frame da câmera."""
    # Fundo preto semi-transparente para legibilidade
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (450, 140), (0, 0, 0), -1) 
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Textos
    texto_item = f"Item: {item}{conf_str}"
    texto_peso = f"Peso: {peso:.5f} kg"
    texto_preco = f"Preco: R$ {preco:.2f}"
    
    # Parâmetros do texto
    font_escala = 1.0
    font_espessura = 2
    cor_texto = (0, 255, 0) # Verde
    pos_item = (10, 40)
    pos_peso = (10, 80)
    pos_preco = (10, 120)

    cv2.putText(frame, texto_item, pos_item, cv2.FONT_HERSHEY_SIMPLEX, font_escala, cor_texto, font_espessura)
    cv2.putText(frame, texto_peso, pos_peso, cv2.FONT_HERSHEY_SIMPLEX, font_escala, cor_texto, font_espessura)
    cv2.putText(frame, texto_preco, pos_preco, cv2.FONT_HERSHEY_SIMPLEX, font_escala, cor_texto, font_espessura)

# --- FIM DAS FUNÇÕES AUXILIARES ---


# --- 4. LOOP PRINCIPAL DA APLICAÇÃO ---

print("\n--- Balança Inteligente Iniciada ---")
print("Pressione 'q' na janela da câmera para sair.")

while True:
    # --- LEITURA DO ARDUINO ---
    peso_lido = ler_peso_da_balanca()
    if peso_lido is not None:
        peso_atual = peso_lido 

    # --- LEITURA DA CÂMERA ---
    ret, frame = cap.read()
    if not ret:
        print("Erro ao ler frame da câmera.")
        break

    # --- PROCESSAMENTO DA IA ---
    # Roda a detecção com o limiar de confiança definido
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False) 
    
    # Desenha as caixas (sempre)
    frame_anotado = results[0].plot()
    
    item_detectado_agora = "Nenhum item"
    confianca_atual = 0.0

    # Pega a detecção com maior confiança
    detections = results[0].boxes.data.cpu().numpy()
    if len(detections) > 0:
        best_detection = detections[detections[:, 4].argmax()]
        confianca_atual = float(best_detection[4])
        class_id = int(best_detection[5])
        item_detectado_agora = model.names[class_id]
        
    # --- MUDANÇA (LÓGICA DE MEDIÇÃO DE TEMPO) ---
    
    # Compara o item deste frame com o do frame anterior
    if item_detectado_agora == item_anterior:
        # É o mesmo item. Verifica se o timer já começou.
        if detection_start_time is None and item_detectado_agora != "Nenhum item":
            # Inicia o timer
            detection_start_time = time.time()
            
        # Se o timer já começou, verifica se atingiu a duração de estabilidade
        elif detection_start_time is not None:
            tempo_decorrido = time.time() - detection_start_time
            
            # Se o item AINDA NÃO está estável, mas ATINGIU o limiar...
            if item_estavel != item_detectado_agora and tempo_decorrido >= STABILITY_DURATION_SEC:
                
                # 1. Trava o novo item
                item_estavel = item_detectado_agora
                confianca_estavel = confianca_atual
                
                # 2. Calcula o preço final com o peso ATUAL
                preco_unitario = PRECOS_POR_KG.get(item_estavel, 0.0)
                preco_total_estavel = peso_atual * preco_unitario
                
                # 3. !! ESTA É A SUA MÉTRICA QUANTITATIVA !!
                # Imprime no terminal o tempo que levou para estabilizar
                print(f"AVALIAÇÃO: Item '{item_estavel}' estabilizado em {tempo_decorrido:.2f} segundos.")

    else:
        # É um item diferente (ou "Nenhum item"). Reseta tudo.
        item_anterior = item_detectado_agora
        detection_start_time = None # Reseta o timer
        
        # Se o item estável foi removido, reseta a GUI
        if item_detectado_agora == "Nenhum item":
            item_estavel = "Nenhum item"
            confianca_estavel = 0.0
            preco_total_estavel = 0.0
            
    # --- FIM DA MUDANÇA ---

    # --- CÁLCULO E EXIBIÇÃO ---
    # Formata a confiança para exibição
    if confianca_estavel > 0:
        confianca_str = f" ({confianca_estavel*100:.0f}%)"
    else:
        confianca_str = ""

    # Desenha na tela os valores ESTÁVEIS
    desenhar_info_na_tela(frame_anotado, item_estavel, peso_atual, preco_total_estavel, confianca_str)
    
    cv2.imshow("Balança Inteligente - IA (Pressione 'q' para sair)", frame_anotado)

    # --- SAÍDA ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. FINALIZAÇÃO ---
cap.release()
cv2.destroyAllWindows()
if arduino and arduino.is_open:
    arduino.close()
print("Aplicação finalizada.")