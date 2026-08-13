import os
import cv2
import math
import time
import queue
import threading
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from deepface import DeepFace


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

CAMERA_INDEX = 0

INTERVALO_ANALISE = 0.35          # segundos entre chamadas ao DeepFace
TAMANHO_DEEPFACE = (160, 160)      # tamanho do rosto enviado à IA
HISTORICO_TAMANHO = 3              # quantidade de análises usadas na suavização

# Pasta onde a logo deve ser colocada (qualquer uma das extensões abaixo).
# Basta salvar o arquivo como "logo.png" (ou .jpg/.jpeg/.webp) nessa pasta.
CAMINHO_LOGO_BASE = r"C:\Users\fabri\OneDrive\Área de Trabalho\SIREN\imgs\logo"
EXTENSOES_LOGO = (".png", ".jpg", ".jpeg", ".webp")


# ============================================================
# LAYOUT DA INTERFACE
# ============================================================

LARGURA_TELA = 1600
ALTURA_TELA = 920

MARGEM = 40
GAP = 30
CONTEUDO_Y = 170
ALTURA_CONTEUDO = 640

CAMERA_LARGURA = 980
CAMERA_ALTURA = ALTURA_CONTEUDO
CAMERA_X = MARGEM
CAMERA_Y = CONTEUDO_Y

PAINEL_X = CAMERA_X + CAMERA_LARGURA + GAP
PAINEL_Y = CONTEUDO_Y
PAINEL_LARGURA = LARGURA_TELA - MARGEM - PAINEL_X
PAINEL_ALTURA = ALTURA_CONTEUDO

RODAPE_Y = CONTEUDO_Y + ALTURA_CONTEUDO + GAP
RODAPE_ALTURA = ALTURA_TELA - RODAPE_Y - 30

RAIO_CARTAO = 26
PADDING_CARTAO = 30

# Espaço reservado no canto superior direito para a logo do IAC.
LOGO_LARGURA_MAX = 150
LOGO_ALTURA_MAX = 86
LOGO_MARGEM_TOPO = 22
LOGO_X2 = LARGURA_TELA - MARGEM
LOGO_X1 = LOGO_X2 - LOGO_LARGURA_MAX
LOGO_Y1 = LOGO_MARGEM_TOPO
LOGO_Y2 = LOGO_Y1 + LOGO_ALTURA_MAX

# Botão "Gerar relatório" no rodapé.
BOTAO_RELATORIO_LARGURA = 300
BOTAO_RELATORIO_ALTURA = 46
BOTAO_RELATORIO_X = (LARGURA_TELA - BOTAO_RELATORIO_LARGURA) // 2
BOTAO_RELATORIO_Y = RODAPE_Y + (RODAPE_ALTURA - BOTAO_RELATORIO_ALTURA) // 2
BOTAO_RELATORIO_RECT = (
    BOTAO_RELATORIO_X,
    BOTAO_RELATORIO_Y,
    BOTAO_RELATORIO_X + BOTAO_RELATORIO_LARGURA,
    BOTAO_RELATORIO_Y + BOTAO_RELATORIO_ALTURA,
)

JANELA_RELATORIO = "SIREN - Relatorio da sessao"

FUNDO_TOPO = (32, 36, 74)
FUNDO_BASE = (72, 78, 138)

BRANCO = (255, 255, 255)
TEXTO_CLARO = (240, 242, 250)
TEXTO_CLARO_SUAVE = (196, 201, 224)
TEXTO_ESCURO = (40, 43, 61)
TEXTO_MUTED = (140, 145, 168)

CARTAO_BORDA = (228, 230, 240)
TRILHA_BARRA = (232, 234, 242)

SUCESSO = (76, 217, 140)
NEUTRO_STATUS = (170, 174, 190)
AO_VIVO = (235, 70, 70)

ACENTO_BOTAO = (108, 99, 255)
ACENTO_BOTAO_ATIVO = (86, 78, 214)


EMOCOES = {
    "happy":    {"label": "ALEGRIA",   "cor": (255, 196, 60)},
    "surprise": {"label": "SURPRESA",  "cor": (247, 150, 66)},
    "neutral":  {"label": "NEUTRO",    "cor": (156, 162, 180)},
    "sad":      {"label": "TRISTEZA",  "cor": (86, 148, 227)},
    "fear":     {"label": "MEDO",      "cor": (159, 108, 217)},
    "disgust":  {"label": "NOJO",      "cor": (108, 191, 130)},
    "angry":    {"label": "RAIVA",     "cor": (232, 77, 77)},
}


def bgr(cor):
    """Converte uma cor RGB para BGR (usado nas funções nativas do OpenCV)."""
    return (cor[2], cor[1], cor[0])


def cor_emocao(nome_ingles):
    return EMOCOES.get(nome_ingles, {}).get("cor", NEUTRO_STATUS)


def label_emocao(nome_ingles):
    return EMOCOES.get(nome_ingles, {}).get("label", nome_ingles.upper())


# ============================================================
# FONTES
# ============================================================

def carregar_fonte(candidatos, tamanho):
    for caminho in candidatos:
        try:
            return ImageFont.truetype(caminho, tamanho)
        except Exception:
            continue
    return ImageFont.load_default()


FONTES_TITULO = [
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/ariblk.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

FONTES_REGULAR = [
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]

FONTES_SEMIBOLD = [
    "C:/Windows/Fonts/segoeuisb.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]

FONTE_TITULO = carregar_fonte(FONTES_TITULO, 62)
FONTE_SUBTITULO = carregar_fonte(FONTES_REGULAR, 22)
FONTE_ROTULO_PEQUENO = carregar_fonte(FONTES_SEMIBOLD, 16)
FONTE_EMOCAO_GRANDE = carregar_fonte(FONTES_SEMIBOLD, 46)
FONTE_CONFIANCA = carregar_fonte(FONTES_REGULAR, 20)
FONTE_BARRA_LABEL = carregar_fonte(FONTES_REGULAR, 18)
FONTE_BARRA_PCT = carregar_fonte(FONTES_SEMIBOLD, 18)
FONTE_RODAPE = carregar_fonte(FONTES_REGULAR, 16)
FONTE_CHIP = carregar_fonte(FONTES_SEMIBOLD, 22)
FONTE_STATUS = carregar_fonte(FONTES_REGULAR, 17)
FONTE_BOTAO = carregar_fonte(FONTES_SEMIBOLD, 18)
FONTE_LOGO_PLACEHOLDER = carregar_fonte(FONTES_REGULAR, 13)


# ============================================================
# DETECTOR DE ROSTOS
# ============================================================

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise RuntimeError("Não foi possível carregar o detector de rostos.")


# ============================================================
# CONCORRÊNCIA: FILA, RESULTADO E HISTÓRICO
# ============================================================

fila_deepface = queue.Queue(maxsize=1)

resultado_ia = None
resultado_lock = threading.Lock()

historico_emocoes = deque(maxlen=HISTORICO_TAMANHO)
historico_lock = threading.Lock()

encerrar_thread = threading.Event()

# Log da sessão atual, usado apenas para montar o relatório em pizza.
# Vive somente em memória: nunca é salvo em disco e é descartado quando o
# programa é fechado (nada fica gravado após o encerramento).
picos_sessao = []
sessao_lock = threading.Lock()

relatorio_aberto = False


# ============================================================
# THREAD DO DEEPFACE
# ============================================================

def trabalhador_deepface():
    global resultado_ia

    print("Thread do DeepFace iniciada (CPU).")

    while not encerrar_thread.is_set():
        try:
            item = fila_deepface.get(timeout=0.2)
        except queue.Empty:
            continue

        if item is None:
            fila_deepface.task_done()
            break

        rosto, coordenadas, rosto_id = item

        try:
            inicio = time.time()

            analise = DeepFace.analyze(
                img_path=rosto,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )

            tempo_processamento = time.time() - inicio

            dados = analise[0] if isinstance(analise, list) else analise
            emocoes_frame = dados["emotion"]

            with historico_lock:
                historico_emocoes.append(emocoes_frame.copy())

                medias = {}
                for nome in emocoes_frame.keys():
                    valores = [h[nome] for h in historico_emocoes if nome in h]
                    if valores:
                        medias[nome] = float(np.mean(valores))

            emocao_ingles = max(medias, key=medias.get)
            confianca = float(medias[emocao_ingles])

            with resultado_lock:
                resultado_ia = {
                    "x": coordenadas[0],
                    "y": coordenadas[1],
                    "w": coordenadas[2],
                    "h": coordenadas[3],
                    "emocao_ingles": emocao_ingles,
                    "emocao": label_emocao(emocao_ingles),
                    "confianca": confianca,
                    "emocoes": medias,
                    "tempo_processamento": tempo_processamento,
                    "rosto_id": rosto_id,
                }

            # Registra este pico (momento em que uma emoção dominante foi
            # calculada) para o relatório da sessão: hora, dia e emoção.
            with sessao_lock:
                picos_sessao.append((datetime.now(), emocao_ingles, confianca))

            print(
                f"DeepFace: {tempo_processamento:.2f}s | "
                f"{label_emocao(emocao_ingles)} {confianca:.1f}%"
            )

        except Exception as erro:
            print("Erro no DeepFace:", erro)

        finally:
            fila_deepface.task_done()

    print("Thread do DeepFace encerrada.")


thread_deepface = threading.Thread(target=trabalhador_deepface, daemon=True)
thread_deepface.start()


# ============================================================
# CÂMERA
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    encerrar_thread.set()
    raise RuntimeError("Não foi possível abrir a câmera.")

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ============================================================
# JANELA
# ============================================================

NOME_JANELA = "SIREN"
cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ============================================================
# LOGO (canto superior direito)
# ============================================================

def carregar_logo():
    for extensao in EXTENSOES_LOGO:
        caminho = CAMINHO_LOGO_BASE + extensao
        if os.path.isfile(caminho):
            try:
                return Image.open(caminho).convert("RGBA")
            except Exception:
                continue
    return None


LOGO_IMG = carregar_logo()


def colar_logo(canvas_rgba):
    """Cola a logo (se existir) redimensionada e centralizada no espaço
    reservado no canto superior direito. Se o arquivo ainda não tiver sido
    colocado na pasta, desenha apenas um espaço reservado discreto."""

    largura_caixa = LOGO_X2 - LOGO_X1
    altura_caixa = LOGO_Y2 - LOGO_Y1

    if LOGO_IMG is not None:
        logo = LOGO_IMG.copy()
        escala = min(largura_caixa / logo.width, altura_caixa / logo.height, 1.0)
        nova_largura = max(1, int(logo.width * escala))
        nova_altura = max(1, int(logo.height * escala))
        logo = logo.resize((nova_largura, nova_altura), Image.LANCZOS)

        x = LOGO_X2 - nova_largura
        y = LOGO_Y1 + (altura_caixa - nova_altura) // 2
        canvas_rgba.alpha_composite(logo, dest=(x, y))
    else:
        desenho = ImageDraw.Draw(canvas_rgba)
        desenho.rounded_rectangle(
            [LOGO_X1, LOGO_Y1, LOGO_X2, LOGO_Y2],
            radius=12,
            outline=(*TEXTO_CLARO_SUAVE, 110),
            width=1,
        )
        texto = "LOGO IAC"
        caixa = desenho.textbbox((0, 0), texto, font=FONTE_LOGO_PLACEHOLDER)
        largura_texto = caixa[2] - caixa[0]
        altura_texto = caixa[3] - caixa[1]
        x = LOGO_X1 + (largura_caixa - largura_texto) // 2
        y = LOGO_Y1 + (altura_caixa - altura_texto) // 2
        desenho.text((x, y), texto, font=FONTE_LOGO_PLACEHOLDER, fill=(*TEXTO_CLARO_SUAVE, 150))


# ============================================================
# FUNÇÕES DE DESENHO — CONSTRUÇÃO DA BASE ESTÁTICA
# ============================================================

def criar_gradiente(largura, altura, cor_topo, cor_base):
    topo = np.array(cor_topo, dtype=np.float32).reshape(1, 1, 3)
    base = np.array(cor_base, dtype=np.float32).reshape(1, 1, 3)
    alfa = np.linspace(0.0, 1.0, altura, dtype=np.float32).reshape(altura, 1, 1)
    grade = topo * (1 - alfa) + base * alfa
    grade = np.repeat(grade, largura, axis=1).astype(np.uint8)
    return grade


def aplicar_sombra(canvas_rgba, x, y, largura, altura, raio, desfoque=14, alfa=95, deslocamento_y=10):
    pad = desfoque + 10
    camada = Image.new("RGBA", (largura + pad * 2, altura + pad * 2), (0, 0, 0, 0))
    desenho = ImageDraw.Draw(camada)
    desenho.rounded_rectangle(
        [pad, pad, pad + largura, pad + altura],
        radius=raio,
        fill=(10, 10, 25, alfa),
    )
    camada = camada.filter(ImageFilter.GaussianBlur(desfoque))
    canvas_rgba.alpha_composite(camada, dest=(x - pad, y - pad + deslocamento_y))


def texto_centralizado(desenho, texto, y, fonte, cor, largura_total=LARGURA_TELA, x_inicio=0):
    caixa = desenho.textbbox((0, 0), texto, font=fonte)
    largura_texto = caixa[2] - caixa[0]
    x = x_inicio + (largura_total - largura_texto) // 2
    desenho.text((x, y), texto, font=fonte, fill=cor)


def construir_base():
    """Monta, uma única vez, todos os elementos estáticos da interface:
    fundo em degradê, cabeçalho, sombras e o cartão do painel lateral."""

    fundo_rgb = criar_gradiente(LARGURA_TELA, ALTURA_TELA, FUNDO_TOPO, FUNDO_BASE)
    canvas = Image.fromarray(fundo_rgb, "RGB").convert("RGBA")
    desenho = ImageDraw.Draw(canvas)

    # Cabeçalho
    texto_centralizado(desenho, "SIREN", 28, FONTE_TITULO, BRANCO)
    texto_centralizado(
        desenho,
        "Sistema de Inteligência Robótica para Regulação do Humor",
        104,
        FONTE_SUBTITULO,
        TEXTO_CLARO_SUAVE,
    )

    # Espaço reservado da logo, no canto superior direito
    colar_logo(canvas)

    # Sombra do cartão da câmera (o vídeo é colado por cima a cada frame)
    aplicar_sombra(canvas, CAMERA_X, CAMERA_Y, CAMERA_LARGURA, CAMERA_ALTURA, RAIO_CARTAO)

    # Sombra + fundo do cartão do painel lateral
    aplicar_sombra(canvas, PAINEL_X, PAINEL_Y, PAINEL_LARGURA, PAINEL_ALTURA, RAIO_CARTAO)
    desenho.rounded_rectangle(
        [PAINEL_X, PAINEL_Y, PAINEL_X + PAINEL_LARGURA, PAINEL_Y + PAINEL_ALTURA],
        radius=RAIO_CARTAO,
        fill=(255, 255, 255, 255),
        outline=(*CARTAO_BORDA, 255),
        width=2,
    )

    # Fundo do rodapé
    desenho.rounded_rectangle(
        [MARGEM, RODAPE_Y, LARGURA_TELA - MARGEM, RODAPE_Y + RODAPE_ALTURA],
        radius=16,
        fill=(255, 255, 255, 22),
    )

    return canvas.convert("RGB")


BASE_PIL = construir_base()
BASE_NP = cv2.cvtColor(np.array(BASE_PIL), cv2.COLOR_RGB2BGR)


# ============================================================
# FUNÇÕES DE DESENHO — CONTEÚDO DINÂMICO (por frame)
# ============================================================

def mascara_arredondada(largura, altura, raio):
    mascara = Image.new("L", (largura, altura), 0)
    ImageDraw.Draw(mascara).rounded_rectangle([0, 0, largura - 1, altura - 1], radius=raio, fill=255)
    return mascara


MASCARA_CAMERA = mascara_arredondada(CAMERA_LARGURA, CAMERA_ALTURA, RAIO_CARTAO)


def colar_camera(tela_pil, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb, "RGB")
    tela_pil.paste(frame_pil, (CAMERA_X, CAMERA_Y), MASCARA_CAMERA)


def desenhar_barra(desenho, x, y, largura, altura, percentual, cor):
    raio = altura // 2
    desenho.rounded_rectangle([x, y, x + largura, y + altura], radius=raio, fill=(*TRILHA_BARRA, 255))

    largura_preenchida = max(altura, int(largura * min(percentual, 100) / 100))
    desenho.rounded_rectangle(
        [x, y, x + largura_preenchida, y + altura], radius=raio, fill=(*cor, 255)
    )


def desenhar_painel(tela_pil, resultado):
    desenho = ImageDraw.Draw(tela_pil)

    px = PAINEL_X + PADDING_CARTAO
    py = PAINEL_Y + PADDING_CARTAO
    largura_util = PAINEL_LARGURA - 2 * PADDING_CARTAO

    # --------------------------------------------------------
    # Sem rosto detectado: estado vazio, acolhedor
    # --------------------------------------------------------
    if resultado is None:
        desenho.text((px, py), "EMOÇÃO DETECTADA", font=FONTE_ROTULO_PEQUENO, fill=TEXTO_MUTED)

        centro_y = PAINEL_Y + PAINEL_ALTURA // 2
        raio_circulo = 46
        cx = PAINEL_X + PAINEL_LARGURA // 2

        desenho.ellipse(
            [cx - raio_circulo, centro_y - 70 - raio_circulo, cx + raio_circulo, centro_y - 70 + raio_circulo],
            outline=(*TEXTO_MUTED, 255),
            width=3,
        )
        # rosto simples e amigável dentro do círculo
        desenho.ellipse([cx - 16, centro_y - 90, cx - 6, centro_y - 80], fill=(*TEXTO_MUTED, 255))
        desenho.ellipse([cx + 6, centro_y - 90, cx + 16, centro_y - 80], fill=(*TEXTO_MUTED, 255))
        desenho.line([cx - 16, centro_y - 62, cx + 16, centro_y - 62], fill=(*TEXTO_MUTED, 255), width=3)

        msg = "Aguardando rosto..."
        caixa = desenho.textbbox((0, 0), msg, font=FONTE_EMOCAO_GRANDE)
        desenho.text(
            (cx - (caixa[2] - caixa[0]) // 2, centro_y + 10),
            msg,
            font=FONTE_BARRA_LABEL,
            fill=TEXTO_ESCURO,
        )
        sub = "Posicione seu rosto em frente à câmera"
        caixa2 = desenho.textbbox((0, 0), sub, font=FONTE_RODAPE)
        desenho.text(
            (cx - (caixa2[2] - caixa2[0]) // 2, centro_y + 42),
            sub,
            font=FONTE_RODAPE,
            fill=TEXTO_MUTED,
        )
        return

    # --------------------------------------------------------
    # Emoção dominante
    # --------------------------------------------------------
    emocoes_ordenadas = sorted(resultado["emocoes"].items(), key=lambda item: item[1], reverse=True)
    dominante_ingles, dominante_valor = emocoes_ordenadas[0]
    cor_dominante = cor_emocao(dominante_ingles)

    desenho.text((px, py), "EMOÇÃO PRINCIPAL", font=FONTE_ROTULO_PEQUENO, fill=TEXTO_MUTED)

    y_nome = py + 26
    desenho.ellipse([px, y_nome + 8, px + 18, y_nome + 26], fill=(*cor_dominante, 255))
    desenho.text((px + 30, y_nome), label_emocao(dominante_ingles), font=FONTE_EMOCAO_GRANDE, fill=TEXTO_ESCURO)

    y_confianca = y_nome + 62
    desenho.text(
        (px + 30, y_confianca),
        f"{dominante_valor:.1f}% de confiança (média suavizada)",
        font=FONTE_CONFIANCA,
        fill=TEXTO_MUTED,
    )

    y_linha = y_confianca + 40
    desenho.line([px, y_linha, px + largura_util, y_linha], fill=(*CARTAO_BORDA, 255), width=2)

    # --------------------------------------------------------
    # Distribuição de todas as emoções
    # --------------------------------------------------------
    y_secao = y_linha + 24
    desenho.text((px, y_secao), "DISTRIBUIÇÃO EMOCIONAL", font=FONTE_ROTULO_PEQUENO, fill=TEXTO_MUTED)

    altura_barra = 14
    altura_linha = 46
    y_barras = y_secao + 34

    largura_label = 118
    largura_pct = 56
    largura_barra = largura_util - largura_label - largura_pct

    for indice, (nome_ingles, valor) in enumerate(emocoes_ordenadas):
        y = y_barras + indice * altura_linha
        cor = cor_emocao(nome_ingles)

        if indice == 0:
            desenho.rounded_rectangle(
                [px - 10, y - 8, px + largura_util + 10, y + altura_linha - 16],
                radius=12,
                fill=(*cor, 24),
            )

        desenho.text((px, y), label_emocao(nome_ingles), font=FONTE_BARRA_LABEL, fill=TEXTO_ESCURO)

        x_barra = px + largura_label
        desenhar_barra(desenho, x_barra, y + 3, largura_barra, altura_barra, valor, cor)

        texto_pct = f"{valor:.0f}%"
        caixa_pct = desenho.textbbox((0, 0), texto_pct, font=FONTE_BARRA_PCT)
        x_pct = px + largura_util - (caixa_pct[2] - caixa_pct[0])
        desenho.text((x_pct, y), texto_pct, font=FONTE_BARRA_PCT, fill=TEXTO_ESCURO)


def desenhar_rodape(tela_pil, fps, resultado):
    desenho = ImageDraw.Draw(tela_pil)
    y_texto = RODAPE_Y + (RODAPE_ALTURA - 18) // 2

    desenho.text((MARGEM + 24, y_texto), "Pressione Q para sair", font=FONTE_RODAPE, fill=TEXTO_CLARO_SUAVE)

    if resultado is not None:
        texto_direita = (
            f"Processamento: {resultado['tempo_processamento'] * 1000:.0f} ms   |   {fps:.0f} FPS"
        )
    else:
        texto_direita = f"{fps:.0f} FPS"

    caixa = desenho.textbbox((0, 0), texto_direita, font=FONTE_RODAPE)
    x_direita = LARGURA_TELA - MARGEM - 24 - (caixa[2] - caixa[0])
    desenho.text((x_direita, y_texto), texto_direita, font=FONTE_RODAPE, fill=TEXTO_CLARO_SUAVE)


def desenhar_botao_relatorio(tela_pil):
    """Botão central do rodapé que abre/fecha o relatório em pizza da sessão."""
    desenho = ImageDraw.Draw(tela_pil)

    x1, y1, x2, y2 = BOTAO_RELATORIO_RECT
    cor = ACENTO_BOTAO_ATIVO if relatorio_aberto else ACENTO_BOTAO

    desenho.rounded_rectangle([x1, y1, x2, y2], radius=y2 - y1, fill=(*cor, 255))

    texto = "OCULTAR RELATÓRIO" if relatorio_aberto else "GERAR RELATÓRIO DA SESSÃO"
    caixa = desenho.textbbox((0, 0), texto, font=FONTE_BOTAO)
    largura_texto = caixa[2] - caixa[0]
    altura_texto = caixa[3] - caixa[1]
    cx = x1 + (x2 - x1 - largura_texto) // 2
    cy = y1 + (y2 - y1 - altura_texto) // 2 - caixa[1]
    desenho.text((cx, cy), texto, font=FONTE_BOTAO, fill=BRANCO)


# ------------------------------------------------------------
# Relatório em pizza (picos de emoção da sessão)
# ------------------------------------------------------------

def figura_para_imagem_bgr(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    imagem_rgb = buffer[:, :, :3]
    return cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR)


def gerar_relatorio_pizza():
    """Gera um gráfico de pizza com os picos de emoção registrados desde que
    o programa foi aberto. Nada é salvo em disco: o gráfico existe apenas em
    memória e some quando a janela do relatório ou o programa é fechado."""

    with sessao_lock:
        dados = list(picos_sessao)

    if not dados:
        aviso = np.full((260, 620, 3), 255, dtype=np.uint8)
        cv2.putText(
            aviso,
            "Ainda nao ha dados suficientes nesta sessao.",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            aviso,
            "Posicione o rosto na camera e tente novamente.",
            (30, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (110, 110, 110),
            2,
            cv2.LINE_AA,
        )
        cv2.namedWindow(JANELA_RELATORIO, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(JANELA_RELATORIO, aviso)
        return

    contagem = {}
    for _, emocao_ingles, _ in dados:
        contagem[emocao_ingles] = contagem.get(emocao_ingles, 0) + 1

    # Ordena do mais frequente para o menos frequente, para leitura mais fácil.
    itens_ordenados = sorted(contagem.items(), key=lambda item: item[1], reverse=True)
    rotulos = [label_emocao(nome) for nome, _ in itens_ordenados]
    valores = [valor for _, valor in itens_ordenados]
    cores = [tuple(c / 255 for c in cor_emocao(nome)) for nome, _ in itens_ordenados]

    total = len(dados)
    inicio = dados[0][0]
    fim = dados[-1][0]
    data_str = inicio.strftime("%d/%m/%Y")
    intervalo_str = f'{inicio.strftime("%H:%M:%S")} - {fim.strftime("%H:%M:%S")}'

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=100)
    fig.patch.set_facecolor("white")

    wedges, textos, autotextos = ax.pie(
        valores,
        labels=rotulos,
        colors=cores,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        pctdistance=0.75,
        textprops={"fontsize": 11, "color": "#28293D"},
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for autotexto in autotextos:
        autotexto.set_color("white")
        autotexto.set_fontweight("bold")

    ax.set_title(
        f"Picos de emoção — {data_str}\n{intervalo_str}   |   {total} análises registradas",
        fontsize=13,
        color="#28293D",
        pad=18,
    )
    ax.axis("equal")
    fig.tight_layout()

    imagem = figura_para_imagem_bgr(fig)
    plt.close(fig)

    cv2.namedWindow(JANELA_RELATORIO, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(JANELA_RELATORIO, imagem)


def ao_clicar_mouse(evento, x, y, flags, param):
    global relatorio_aberto

    if evento != cv2.EVENT_LBUTTONDOWN:
        return

    # Corrige a posição do clique caso a janela esteja em tela cheia numa
    # resolução diferente da resolução interna da interface (1600x920).
    try:
        _, _, largura_janela, altura_janela = cv2.getWindowImageRect(NOME_JANELA)
    except Exception:
        largura_janela, altura_janela = LARGURA_TELA, ALTURA_TELA

    escala_x = LARGURA_TELA / largura_janela if largura_janela else 1.0
    escala_y = ALTURA_TELA / altura_janela if altura_janela else 1.0
    x_real = x * escala_x
    y_real = y * escala_y

    bx1, by1, bx2, by2 = BOTAO_RELATORIO_RECT
    if bx1 <= x_real <= bx2 and by1 <= y_real <= by2:
        if relatorio_aberto:
            cv2.destroyWindow(JANELA_RELATORIO)
            relatorio_aberto = False
        else:
            gerar_relatorio_pizza()
            relatorio_aberto = True


cv2.setMouseCallback(NOME_JANELA, ao_clicar_mouse)


# ------------------------------------------------------------
# Elementos desenhados sobre o frame da câmera (espaço BGR)
# ------------------------------------------------------------

def retangulo_arredondado_cv2(imagem, pt1, pt2, raio, cor):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(imagem, (x1 + raio, y1), (x2 - raio, y2), cor, -1)
    cv2.rectangle(imagem, (x1, y1 + raio), (x2, y2 - raio), cor, -1)
    cv2.circle(imagem, (x1 + raio, y1 + raio), raio, cor, -1)
    cv2.circle(imagem, (x2 - raio, y1 + raio), raio, cor, -1)
    cv2.circle(imagem, (x1 + raio, y2 - raio), raio, cor, -1)
    cv2.circle(imagem, (x2 - raio, y2 - raio), raio, cor, -1)


def desenhar_marcadores_rosto(frame, x, y, w, h, cor_bgr):
    comprimento = int(min(w, h) * 0.22)
    espessura = 4

    pontos = [
        ((x, y), (1, 0), (0, 1)),
        ((x + w, y), (-1, 0), (0, 1)),
        ((x, y + h), (1, 0), (0, -1)),
        ((x + w, y + h), (-1, 0), (0, -1)),
    ]

    for canto, direcao_x, direcao_y in pontos:
        cx, cy = canto
        cv2.line(frame, (cx, cy), (cx + direcao_x[0] * comprimento, cy), cor_bgr, espessura)
        cv2.line(frame, (cx, cy), (cx, cy + direcao_y[1] * comprimento), cor_bgr, espessura)


def desenhar_chip_emocao(frame, x, y, texto, cor_bgr):
    fonte_cv = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.8
    espessura = 2
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte_cv, escala, espessura)

    pad_x, pad_y = 16, 10
    x1, y1 = x, max(y - altura_texto - pad_y * 2 - 12, 10)
    x2, y2 = x + largura_texto + pad_x * 2, y1 + altura_texto + pad_y * 2

    retangulo_arredondado_cv2(frame, (x1, y1), (x2, y2), 12, cor_bgr)
    cv2.putText(
        frame,
        texto,
        (x1 + pad_x, y2 - pad_y - 4),
        fonte_cv,
        escala,
        bgr(BRANCO),
        espessura,
        cv2.LINE_AA,
    )


def desenhar_status_camera(frame, rosto_detectado, contador):
    # Indicador "AO VIVO" com leve pulsação
    pulso = 5 + int(2 * (0.5 + 0.5 * math.sin(time.time() * 3)))
    cv2.circle(frame, (34, 34), pulso, bgr(AO_VIVO), -1)
    cv2.putText(frame, "AO VIVO", (52, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr(BRANCO), 2, cv2.LINE_AA)

    # Indicador de status de detecção, no canto superior direito
    largura_frame = frame.shape[1]
    cor_status = SUCESSO if rosto_detectado else NEUTRO_STATUS
    texto_status = "ROSTO DETECTADO" if rosto_detectado else "PROCURANDO ROSTO..."

    (largura_texto, altura_texto), _ = cv2.getTextSize(
        texto_status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    x_texto = largura_frame - largura_texto - 46
    cv2.circle(frame, (largura_frame - 30, 34), 8, bgr(cor_status), -1)
    cv2.putText(
        frame, texto_status, (x_texto, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, bgr(BRANCO), 2, cv2.LINE_AA
    )


# ============================================================
# INFORMAÇÕES DE INICIALIZAÇÃO
# ============================================================

print()
print("==============================================")
print("                  SIREN")
print("==============================================")
print("Sistema de Inteligência Robótica para Regulação do Humor")
print("==============================================")
print("Câmera iniciada | DeepFace: CPU | GPU: DESATIVADA")
print(f"Entrada DeepFace: {TAMANHO_DEEPFACE[0]}x{TAMANHO_DEEPFACE[1]}")
print(f"Intervalo de análise: {INTERVALO_ANALISE:.2f}s | Suavização: {HISTORICO_TAMANHO} análises")
if LOGO_IMG is None:
    print(f"Logo não encontrada em: {CAMINHO_LOGO_BASE}.<png/jpg/jpeg/webp>")
print("Pressione Q para sair. Clique no botão do rodapé para ver o relatório da sessão.")
print("==============================================")
print()


# ============================================================
# LOOP PRINCIPAL
# ============================================================

ultimo_rosto = None
ultimo_tempo_analise = 0
id_rosto = 0
rosto_detectado_anteriormente = False

tempos_frame = deque(maxlen=30)

try:
    while True:
        inicio_frame = time.time()

        sucesso, frame = camera.read()
        if not sucesso:
            continue

        frame = cv2.flip(frame, 1)

        # --------------------------------------------------
        # Detecção de rosto (em resolução reduzida, por performance)
        # --------------------------------------------------
        pequeno = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cinza = cv2.equalizeHist(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY))

        rostos_pequenos = face_cascade.detectMultiScale(
            cinza, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        maior_rosto = None
        maior_area = 0
        for (x, y, w, h) in rostos_pequenos:
            x, y, w, h = x * 2, y * 2, w * 2, h * 2
            area = w * h
            if area > maior_area:
                maior_area = area
                maior_rosto = (x, y, w, h)

        rosto_detectado = maior_rosto is not None

        if rosto_detectado:
            rosto_detectado_anteriormente = True
            x, y, w, h = maior_rosto

            margem = int(max(w, h) * 0.15)
            x1 = max(0, x - margem)
            y1 = max(0, y - margem)
            x2 = min(frame.shape[1], x + w + margem)
            y2 = min(frame.shape[0], y + h + margem)

            rosto = frame[y1:y2, x1:x2]

            if rosto.size > 0:
                rosto_deepface = cv2.resize(rosto, TAMANHO_DEEPFACE, interpolation=cv2.INTER_AREA)
                coordenadas = (x1, y1, x2 - x1, y2 - y1)
                ultimo_rosto = (rosto_deepface.copy(), coordenadas, id_rosto)

                agora = time.time()
                if agora - ultimo_tempo_analise >= INTERVALO_ANALISE:
                    ultimo_tempo_analise = agora
                    try:
                        fila_deepface.put_nowait(ultimo_rosto)
                    except queue.Full:
                        try:
                            fila_deepface.get_nowait()
                            fila_deepface.task_done()
                        except queue.Empty:
                            pass
                        try:
                            fila_deepface.put_nowait(ultimo_rosto)
                        except queue.Full:
                            pass

            id_rosto += 1
        else:
            if rosto_detectado_anteriormente:
                rosto_detectado_anteriormente = False
                with historico_lock:
                    historico_emocoes.clear()
                with resultado_lock:
                    resultado_ia = None

        # --------------------------------------------------
        # Resultado mais recente da IA
        # --------------------------------------------------
        with resultado_lock:
            resultado_atual = resultado_ia

        # --------------------------------------------------
        # Overlays sobre o vídeo (marcadores + chip + status)
        # --------------------------------------------------
        if resultado_atual is not None and rosto_detectado:
            rx, ry = resultado_atual["x"], resultado_atual["y"]
            rw, rh = resultado_atual["w"], resultado_atual["h"]
            cor_atual = bgr(cor_emocao(resultado_atual["emocao_ingles"]))

            desenhar_marcadores_rosto(frame, rx, ry, rw, rh, cor_atual)
            desenhar_chip_emocao(
                frame,
                rx,
                ry,
                f"{resultado_atual['emocao']}  {resultado_atual['confianca']:.0f}%",
                cor_atual,
            )

        desenhar_status_camera(frame, rosto_detectado, id_rosto)

        # --------------------------------------------------
        # Montagem da interface final
        # --------------------------------------------------
        frame_camera = cv2.resize(frame, (CAMERA_LARGURA, CAMERA_ALTURA), interpolation=cv2.INTER_AREA)

        tempos_frame.append(time.time() - inicio_frame)
        fps = 1.0 / (sum(tempos_frame) / len(tempos_frame)) if tempos_frame else 0.0

        tela_pil = BASE_PIL.copy()
        colar_camera(tela_pil, frame_camera)
        desenhar_painel(tela_pil, resultado_atual)
        desenhar_rodape(tela_pil, fps, resultado_atual)
        desenhar_botao_relatorio(tela_pil)

        tela_final = cv2.cvtColor(np.array(tela_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow(NOME_JANELA, tela_final)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break

except KeyboardInterrupt:
    print()
    print("Interrupção pelo usuário.")

finally:
    print()
    print("Encerrando SIREN...")

    encerrar_thread.set()

    try:
        fila_deepface.put_nowait(None)
    except queue.Full:
        try:
            fila_deepface.get_nowait()
            fila_deepface.task_done()
            fila_deepface.put_nowait(None)
        except queue.Empty:
            pass

    camera.release()
    cv2.destroyAllWindows()
    thread_deepface.join(timeout=3)

    # Apaga os dados da sessão da memória: nada do relatório fica salvo
    # após o programa ser fechado.
    with sessao_lock:
        picos_sessao.clear()

    print("SIREN encerrado.")
