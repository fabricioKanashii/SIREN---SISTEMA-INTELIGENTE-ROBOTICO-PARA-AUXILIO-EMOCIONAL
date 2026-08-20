import queue
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pygame
from deepface import DeepFace

from config import (
    ConfiguracaoGeral,
    Emocoes
)


# ============================================================
# DETECTOR DE ROSTO
# ============================================================

class DetectorRosto:
    """Detecta o maior rosto presente no frame."""

    def __init__(self):

        caminho_cascade = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self._cascade = cv2.CascadeClassifier(
            caminho_cascade
        )

        if self._cascade.empty():

            raise RuntimeError(
                "Não foi possível carregar o Haar Cascade."
            )

    def detectar_maior_rosto(self, frame_bgr):

        if frame_bgr is None:
            return None

        if frame_bgr.size == 0:
            return None

        pequeno = cv2.resize(
            frame_bgr,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_AREA
        )

        cinza = cv2.cvtColor(
            pequeno,
            cv2.COLOR_BGR2GRAY
        )

        cinza = cv2.equalizeHist(
            cinza
        )

        rostos = self._cascade.detectMultiScale(
            cinza,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        maior_rosto = None
        maior_area = 0

        for (x, y, w, h) in rostos:

            x *= 2
            y *= 2
            w *= 2
            h *= 2

            area = w * h

            if area > maior_area:

                maior_area = area

                maior_rosto = (
                    x,
                    y,
                    w,
                    h
                )

        return maior_rosto


# ============================================================
# SESSÃO
# ============================================================

class GerenciadorSessao:
    """Armazena os resultados emocionais da sessão."""

    def __init__(self):

        self._picos = []

        self._lock = threading.Lock()

    def registrar(
        self,
        emocao_ingles,
        confianca
    ):

        with self._lock:

            self._picos.append(
                (
                    datetime.now(),
                    emocao_ingles,
                    confianca
                )
            )

    def obter_dados(self):

        with self._lock:

            return list(
                self._picos
            )

    def limpar(self):

        with self._lock:

            self._picos.clear()


# ============================================================
# ANALISADOR DE EMOÇÕES
# ============================================================

class AnalisadorEmocoes:
    """
    Executa o DeepFace em uma thread separada.
    """

    def __init__(
        self,
        gerenciador_sessao
    ):

        self._gerenciador_sessao = (
            gerenciador_sessao
        )

        self._fila = queue.Queue(
            maxsize=1
        )

        self._resultado = None

        self._resultado_lock = (
            threading.Lock()
        )

        self._historico = deque(
            maxlen=ConfiguracaoGeral.HISTORICO_TAMANHO
        )

        self._historico_lock = (
            threading.Lock()
        )

        self._encerrar = (
            threading.Event()
        )

        self._thread = threading.Thread(
            target=self._executar,
            daemon=True,
            name="SIREN-DeepFace"
        )

        self._iniciada = False

    # --------------------------------------------------------
    # INICIAR
    # --------------------------------------------------------

    def iniciar(self):

        if self._iniciada:
            return

        self._iniciada = True

        self._thread.start()

        print(
            "[SIREN] Thread do DeepFace iniciada."
        )

    # --------------------------------------------------------
    # ENCERRAR
    # --------------------------------------------------------

    def encerrar(self):

        if not self._iniciada:
            return

        self._encerrar.set()

        try:

            self._fila.put_nowait(
                None
            )

        except queue.Full:

            try:

                item = self._fila.get_nowait()

                self._fila.task_done()

                del item

                self._fila.put_nowait(
                    None
                )

            except queue.Empty:
                pass

        if self._thread.is_alive():

            self._thread.join(
                timeout=5
            )

        print(
            "[SIREN] Thread do DeepFace encerrada."
        )

    # --------------------------------------------------------
    # ENVIAR ROSTO
    # --------------------------------------------------------

    def enviar_rosto(
        self,
        rosto,
        coordenadas,
        rosto_id
    ):

        item = (
            rosto,
            coordenadas,
            rosto_id
        )

        try:

            self._fila.put_nowait(
                item
            )

        except queue.Full:

            try:

                antigo = (
                    self._fila.get_nowait()
                )

                self._fila.task_done()

                del antigo

            except queue.Empty:
                pass

            try:

                self._fila.put_nowait(
                    item
                )

            except queue.Full:
                pass

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    def obter_resultado(self):

        with self._resultado_lock:

            return self._resultado

    def limpar_resultado(self):

        with self._resultado_lock:

            self._resultado = None

        with self._historico_lock:

            self._historico.clear()

    # --------------------------------------------------------
    # THREAD
    # --------------------------------------------------------

    def _executar(self):

        while not self._encerrar.is_set():

            try:

                item = self._fila.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            if item is None:

                self._fila.task_done()

                break

            rosto, coordenadas, rosto_id = item

            try:

                self._processar(
                    rosto,
                    coordenadas,
                    rosto_id
                )

            except Exception as erro:

                print(
                    "[DeepFace] Erro:",
                    erro
                )

            finally:

                self._fila.task_done()

    # --------------------------------------------------------
    # PROCESSAMENTO
    # --------------------------------------------------------

    def _processar(
        self,
        rosto,
        coordenadas,
        rosto_id
    ):

        inicio = time.time()

        analise = DeepFace.analyze(
            img_path=rosto,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="skip",
            silent=True
        )

        tempo_processamento = (
            time.time() - inicio
        )

        if isinstance(
            analise,
            list
        ):

            if not analise:
                return

            dados = analise[0]

        else:

            dados = analise

        emocoes_frame = dados.get(
            "emotion",
            {}
        )

        if not emocoes_frame:
            return

        medias = self._suavizar(
            emocoes_frame
        )

        if not medias:
            return

        emocao_ingles = max(
            medias,
            key=medias.get
        )

        confianca = float(
            medias[emocao_ingles]
        )

        resultado = {

            "x": coordenadas[0],
            "y": coordenadas[1],

            "w": coordenadas[2],
            "h": coordenadas[3],

            "emocao_ingles": (
                emocao_ingles
            ),

            "emocao": Emocoes.label(
                emocao_ingles
            ),

            "confianca": confianca,

            "emocoes": medias,

            "tempo_processamento": (
                tempo_processamento
            ),

            "rosto_id": rosto_id
        }

        with self._resultado_lock:

            self._resultado = resultado

        self._gerenciador_sessao.registrar(
            emocao_ingles,
            confianca
        )

        print(
            f"[DeepFace] "
            f"{tempo_processamento:.2f}s | "
            f"{Emocoes.label(emocao_ingles)} | "
            f"{confianca:.1f}%"
        )

    # --------------------------------------------------------
    # SUAVIZAÇÃO
    # --------------------------------------------------------

    def _suavizar(
        self,
        emocoes_frame
    ):

        with self._historico_lock:

            self._historico.append(
                dict(emocoes_frame)
            )

            medias = {}

            for nome in emocoes_frame.keys():

                valores = [
                    h[nome]
                    for h in self._historico
                    if nome in h
                ]

                if valores:

                    medias[nome] = float(
                        np.mean(valores)
                    )

        return medias


# ============================================================
# CÂMERA
# ============================================================

class CameraSiren:
    """Gerencia a câmera do SIREN."""

    def __init__(
        self,
        indice=ConfiguracaoGeral.CAMERA_INDEX
    ):

        self._captura = cv2.VideoCapture(
            indice
        )

        if not self._captura.isOpened():

            raise RuntimeError(
                "Não foi possível abrir a câmera."
            )

        self._captura.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            ConfiguracaoGeral.CAMERA_LARGURA
        )

        self._captura.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            ConfiguracaoGeral.CAMERA_ALTURA
        )

        self._captura.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

    def ler_frame(self):

        sucesso, frame = (
            self._captura.read()
        )

        if not sucesso:
            return False, None

        frame = cv2.flip(
            frame,
            1
        )

        return True, frame

    def liberar(self):

        if self._captura is not None:

            self._captura.release()

            self._captura = None


# ============================================================
# MOTOR SIREN
# ============================================================

class MotorSiren:
    """
    Fachada do backend.
    """

    def __init__(self):

        self.sessao = (
            GerenciadorSessao()
        )

        self.analisador = (
            AnalisadorEmocoes(
                self.sessao
            )
        )

        self.detector = (
            DetectorRosto()
        )

        self.camera = (
            CameraSiren()
        )

        self._ultimo_tempo_analise = 0

        self._id_rosto = 0

        self._rosto_detectado_anteriormente = (
            False
        )

        self._iniciado = False

    # --------------------------------------------------------
    # INICIAR
    # --------------------------------------------------------

    def iniciar(self):

        if self._iniciado:
            return

        self._iniciado = True

        self.analisador.iniciar()

    # --------------------------------------------------------
    # ENCERRAR
    # --------------------------------------------------------

    def encerrar(self):

        try:

            self.analisador.encerrar()

        finally:

            self.camera.liberar()

            self.sessao.limpar()

            self._iniciado = False

    # --------------------------------------------------------
    # PROCESSAR FRAME
    # --------------------------------------------------------

    def processar_frame(
        self,
        frame
    ):

        maior_rosto = (
            self.detector
            .detectar_maior_rosto(frame)
        )

        rosto_detectado = (
            maior_rosto is not None
        )

        coordenadas_expandidas = None

        if rosto_detectado:

            self._rosto_detectado_anteriormente = (
                True
            )

            x, y, w, h = maior_rosto

            margem = int(
                max(w, h) * 0.15
            )

            x1 = max(
                0,
                x - margem
            )

            y1 = max(
                0,
                y - margem
            )

            x2 = min(
                frame.shape[1],
                x + w + margem
            )

            y2 = min(
                frame.shape[0],
                y + h + margem
            )

            rosto_recorte = frame[
                y1:y2,
                x1:x2
            ]

            if rosto_recorte.size > 0:

                rosto_deepface = cv2.resize(
                    rosto_recorte,
                    ConfiguracaoGeral.TAMANHO_DEEPFACE,
                    interpolation=cv2.INTER_AREA
                )

                coordenadas_expandidas = (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )

                agora = time.time()

                if (
                    agora
                    - self._ultimo_tempo_analise
                    >= ConfiguracaoGeral.INTERVALO_ANALISE
                ):

                    self._ultimo_tempo_analise = (
                        agora
                    )

                    self.analisador.enviar_rosto(
                        rosto_deepface.copy(),
                        coordenadas_expandidas,
                        self._id_rosto
                    )

            self._id_rosto += 1

        else:

            if self._rosto_detectado_anteriormente:

                self._rosto_detectado_anteriormente = (
                    False
                )

                self.analisador.limpar_resultado()

        return (
            rosto_detectado,
            coordenadas_expandidas
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    def obter_resultado_ia(self):

        return self.analisador.obter_resultado()

    # --------------------------------------------------------
    # SESSÃO
    # --------------------------------------------------------

    def obter_dados_sessao(self):

        return self.sessao.obter_dados()

# --------------------------------------------------------
# GERENCIADOR DE AUDIO DO SIREN
# --------------------------------------------------------


class AudioSiren:

    def __init__(self):
        pygame.mixer.init()

    def reproduzir_e_esperar(self, caminho):
        pygame.mixer.music.load(caminho)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    def pausar(self):
        pygame.mixer.music.pause()

    def continuar(self):
        pygame.mixer.music.unpause()

    def parar(self):
        pygame.mixer.music.stop()

    def esta_reproduzindo(self):
        return pygame.mixer.music.get_busy()

    def volume(self, valor):
        pygame.mixer.music.set_volume(valor)

    def finalizar(self):
        pygame.mixer.quit()

