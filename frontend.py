import math
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from PIL import (
    Image,
    ImageDraw,
    ImageFilter
)

from matplotlib.backends.backend_agg import (
    FigureCanvasAgg
)

from config import (
    ConfiguracaoGeral,
    Layout,
    Cores,
    Emocoes,
    Fontes
)


# ============================================================
# LOGO
# ============================================================

class LogoSiren:

    def __init__(self):

        self.imagem = self._carregar()

    def _carregar(self):

        for extensao in (
            ConfiguracaoGeral.EXTENSOES_LOGO
        ):

            caminho = (
                ConfiguracaoGeral.CAMINHO_LOGO_BASE
                + extensao
            )

            if os.path.isfile(caminho):

                try:

                    return Image.open(
                        caminho
                    ).convert("RGBA")

                except Exception:
                    continue

        return None

    def colar(
        self,
        canvas_rgba,
        fontes
    ):

        largura = (
            Layout.LOGO_X2
            - Layout.LOGO_X1
        )

        altura = (
            Layout.LOGO_Y2
            - Layout.LOGO_Y1
        )

        if self.imagem is not None:

            self._colar_imagem(
                canvas_rgba,
                largura,
                altura
            )

        else:

            self._colar_placeholder(
                canvas_rgba,
                largura,
                altura,
                fontes
            )

    def _colar_imagem(
        self,
        canvas_rgba,
        largura,
        altura
    ):

        logo = self.imagem.copy()

        escala = min(
            largura / logo.width,
            altura / logo.height,
            1.0
        )

        nova_largura = max(
            1,
            int(logo.width * escala)
        )

        nova_altura = max(
            1,
            int(logo.height * escala)
        )

        logo = logo.resize(
            (
                nova_largura,
                nova_altura
            ),
            Image.LANCZOS
        )

        x = (
            Layout.LOGO_X2
            - nova_largura
        )

        y = (
            Layout.LOGO_Y1
            + (
                altura
                - nova_altura
            ) // 2
        )

        canvas_rgba.alpha_composite(
            logo,
            dest=(x, y)
        )

    def _colar_placeholder(
        self,
        canvas_rgba,
        largura,
        altura,
        fontes
    ):

        desenho = ImageDraw.Draw(
            canvas_rgba
        )

        desenho.rounded_rectangle(
            [
                Layout.LOGO_X1,
                Layout.LOGO_Y1,
                Layout.LOGO_X2,
                Layout.LOGO_Y2
            ],
            radius=12,
            outline=(
                *Cores.TEXTO_CLARO_SUAVE,
                110
            ),
            width=1
        )

        texto = "SIREN"

        caixa = desenho.textbbox(
            (0, 0),
            texto,
            font=fontes.logo_placeholder
        )

        largura_texto = (
            caixa[2] - caixa[0]
        )

        altura_texto = (
            caixa[3] - caixa[1]
        )

        x = (
            Layout.LOGO_X1
            + (
                largura
                - largura_texto
            ) // 2
        )

        y = (
            Layout.LOGO_Y1
            + (
                altura
                - altura_texto
            ) // 2
        )

        desenho.text(
            (x, y),
            texto,
            font=fontes.logo_placeholder,
            fill=(
                *Cores.TEXTO_CLARO_SUAVE,
                150
            )
        )


# ============================================================
# DESENHO
# ============================================================

class Desenho:

    @staticmethod
    def gradiente(
        largura,
        altura,
        cor_topo,
        cor_base
    ):

        topo = np.array(
            cor_topo,
            dtype=np.float32
        ).reshape(
            1,
            1,
            3
        )

        base = np.array(
            cor_base,
            dtype=np.float32
        ).reshape(
            1,
            1,
            3
        )

        alfa = np.linspace(
            0.0,
            1.0,
            altura,
            dtype=np.float32
        ).reshape(
            altura,
            1,
            1
        )

        grade = (
            topo * (1 - alfa)
            + base * alfa
        )

        grade = np.repeat(
            grade,
            largura,
            axis=1
        )

        return grade.astype(
            np.uint8
        )

    @staticmethod
    def sombra(
        canvas_rgba,
        x,
        y,
        largura,
        altura,
        raio,
        desfoque=14,
        alfa=95,
        deslocamento_y=10
    ):

        pad = (
            desfoque
            + 10
        )

        camada = Image.new(
            "RGBA",
            (
                largura + pad * 2,
                altura + pad * 2
            ),
            (
                0,
                0,
                0,
                0
            )
        )

        desenho = ImageDraw.Draw(
            camada
        )

        desenho.rounded_rectangle(
            [
                pad,
                pad,
                pad + largura,
                pad + altura
            ],
            radius=raio,
            fill=(
                10,
                10,
                25,
                alfa
            )
        )

        camada = camada.filter(
            ImageFilter.GaussianBlur(
                desfoque
            )
        )

        canvas_rgba.alpha_composite(
            camada,
            dest=(
                x - pad,
                y - pad + deslocamento_y
            )
        )

    @staticmethod
    def texto_centralizado(
        desenho,
        texto,
        y,
        fonte,
        cor,
        largura_total=Layout.LARGURA_TELA,
        x_inicio=0
    ):

        caixa = desenho.textbbox(
            (0, 0),
            texto,
            font=fonte
        )

        largura_texto = (
            caixa[2] - caixa[0]
        )

        x = (
            x_inicio
            + (
                largura_total
                - largura_texto
            ) // 2
        )

        desenho.text(
            (x, y),
            texto,
            font=fonte,
            fill=cor
        )

    @staticmethod
    def mascara_arredondada(
        largura,
        altura,
        raio
    ):

        mascara = Image.new(
            "L",
            (
                largura,
                altura
            ),
            0
        )

        ImageDraw.Draw(
            mascara
        ).rounded_rectangle(
            [
                0,
                0,
                largura - 1,
                altura - 1
            ],
            radius=raio,
            fill=255
        )

        return mascara

    @staticmethod
    def barra(
        desenho,
        x,
        y,
        largura,
        altura,
        percentual,
        cor
    ):

        raio = altura // 2

        desenho.rounded_rectangle(
            [
                x,
                y,
                x + largura,
                y + altura
            ],
            radius=raio,
            fill=(
                *Cores.TRILHA_BARRA,
                255
            )
        )

        percentual = max(
            0,
            min(
                float(percentual),
                100
            )
        )

        if percentual <= 0:
            return

        largura_preenchida = max(
            altura,
            int(
                largura
                * percentual
                / 100
            )
        )

        largura_preenchida = min(
            largura,
            largura_preenchida
        )

        desenho.rounded_rectangle(
            [
                x,
                y,
                x + largura_preenchida,
                y + altura
            ],
            radius=raio,
            fill=(
                *cor,
                255
            )
        )

    @staticmethod
    def retangulo_arredondado_cv2(
        imagem,
        pt1,
        pt2,
        raio,
        cor
    ):

        x1, y1 = pt1
        x2, y2 = pt2

        cv2.rectangle(
            imagem,
            (
                x1 + raio,
                y1
            ),
            (
                x2 - raio,
                y2
            ),
            cor,
            -1
        )

        cv2.rectangle(
            imagem,
            (
                x1,
                y1 + raio
            ),
            (
                x2,
                y2 - raio
            ),
            cor,
            -1
        )

        cv2.circle(
            imagem,
            (
                x1 + raio,
                y1 + raio
            ),
            raio,
            cor,
            -1
        )

        cv2.circle(
            imagem,
            (
                x2 - raio,
                y1 + raio
            ),
            raio,
            cor,
            -1
        )

        cv2.circle(
            imagem,
            (
                x1 + raio,
                y2 - raio
            ),
            raio,
            cor,
            -1
        )

        cv2.circle(
            imagem,
            (
                x2 - raio,
                y2 - raio
            ),
            raio,
            cor,
            -1
        )


# ============================================================
# PAINEL DE EMOÇÕES
# ============================================================

class PainelEmocoes:

    def __init__(
        self,
        fontes
    ):

        self.fontes = fontes

    def desenhar(
        self,
        tela_pil,
        resultado
    ):

        desenho = ImageDraw.Draw(
            tela_pil
        )

        px = (
            Layout.PAINEL_X
            + Layout.PADDING_CARTAO
        )

        py = (
            Layout.PAINEL_Y
            + Layout.PADDING_CARTAO
        )

        largura_util = (
            Layout.PAINEL_LARGURA
            - 2 * Layout.PADDING_CARTAO
        )

        if resultado is None:

            self._desenhar_vazio(
                desenho,
                px,
                py
            )

            return

        self._desenhar_com_resultado(
            desenho,
            px,
            py,
            largura_util,
            resultado
        )

    def _desenhar_vazio(
        self,
        desenho,
        px,
        py
    ):

        desenho.text(
            (
                px,
                py
            ),
            "EMOÇÃO DETECTADA",
            font=self.fontes.rotulo_pequeno,
            fill=Cores.TEXTO_MUTED
        )

        centro_y = (
            Layout.PAINEL_Y
            + Layout.PAINEL_ALTURA // 2
        )

        cx = (
            Layout.PAINEL_X
            + Layout.PAINEL_LARGURA // 2
        )

        raio = 46

        desenho.ellipse(
            [
                cx - raio,
                centro_y - 70 - raio,
                cx + raio,
                centro_y - 70 + raio
            ],
            outline=(
                *Cores.TEXTO_MUTED,
                255
            ),
            width=3
        )

        desenho.ellipse(
            [
                cx - 16,
                centro_y - 90,
                cx - 6,
                centro_y - 80
            ],
            fill=(
                *Cores.TEXTO_MUTED,
                255
            )
        )

        desenho.ellipse(
            [
                cx + 6,
                centro_y - 90,
                cx + 16,
                centro_y - 80
            ],
            fill=(
                *Cores.TEXTO_MUTED,
                255
            )
        )

        desenho.line(
            [
                cx - 16,
                centro_y - 62,
                cx + 16,
                centro_y - 62
            ],
            fill=(
                *Cores.TEXTO_MUTED,
                255
            ),
            width=3
        )

        msg = "Aguardando rosto..."

        caixa = desenho.textbbox(
            (0, 0),
            msg,
            font=self.fontes.barra_label
        )

        desenho.text(
            (
                cx
                - (
                    caixa[2]
                    - caixa[0]
                ) // 2,
                centro_y + 10
            ),
            msg,
            font=self.fontes.barra_label,
            fill=Cores.TEXTO_ESCURO
        )

        sub = (
            "Posicione seu rosto "
            "em frente à câmera"
        )

        caixa2 = desenho.textbbox(
            (0, 0),
            sub,
            font=self.fontes.rodape
        )

        desenho.text(
            (
                cx
                - (
                    caixa2[2]
                    - caixa2[0]
                ) // 2,
                centro_y + 42
            ),
            sub,
            font=self.fontes.rodape,
            fill=Cores.TEXTO_MUTED
        )

    def _desenhar_com_resultado(
        self,
        desenho,
        px,
        py,
        largura_util,
        resultado
    ):

        emocoes_ordenadas = sorted(
            resultado["emocoes"].items(),
            key=lambda item: item[1],
            reverse=True
        )

        dominante_ingles = (
            emocoes_ordenadas[0][0]
        )

        cor_dominante = (
            Emocoes.cor(
                dominante_ingles
            )
        )

        desenho.text(
            (
                px,
                py
            ),
            "EMOÇÃO PRINCIPAL",
            font=self.fontes.rotulo_pequeno,
            fill=Cores.TEXTO_MUTED
        )

        y_nome = py + 26

        desenho.ellipse(
            [
                px,
                y_nome + 8,
                px + 18,
                y_nome + 26
            ],
            fill=(
                *cor_dominante,
                255
            )
        )

        desenho.text(
            (
                px + 30,
                y_nome
            ),
            Emocoes.label(
                dominante_ingles
            ),
            font=self.fontes.emocao_grande,
            fill=Cores.TEXTO_ESCURO
        )

        y_confianca = (
            y_nome + 62
        )

        desenho.text(
            (
                px + 30,
                y_confianca
            ),
            f"{resultado['confianca']:.1f}% de confiança",
            font=self.fontes.confianca,
            fill=Cores.TEXTO_MUTED
        )

        y_linha = (
            y_confianca + 40
        )

        desenho.line(
            [
                px,
                y_linha,
                px + largura_util,
                y_linha
            ],
            fill=(
                *Cores.CARTAO_BORDA,
                255
            ),
            width=2
        )

        self._desenhar_distribuicao(
            desenho,
            px,
            y_linha,
            largura_util,
            emocoes_ordenadas
        )

    def _desenhar_distribuicao(
        self,
        desenho,
        px,
        y_linha,
        largura_util,
        emocoes_ordenadas
    ):

        y_secao = (
            y_linha + 24
        )

        desenho.text(
            (
                px,
                y_secao
            ),
            "DISTRIBUIÇÃO EMOCIONAL",
            font=self.fontes.rotulo_pequeno,
            fill=Cores.TEXTO_MUTED
        )

        altura_barra = 14
        altura_linha = 46

        y_barras = (
            y_secao + 34
        )

        largura_label = 118
        largura_pct = 56

        largura_barra = (
            largura_util
            - largura_label
            - largura_pct
        )

        for indice, (
            nome_ingles,
            valor
        ) in enumerate(
            emocoes_ordenadas
        ):

            y = (
                y_barras
                + indice
                * altura_linha
            )

            cor = Emocoes.cor(
                nome_ingles
            )

            if indice == 0:

                desenho.rounded_rectangle(
                    [
                        px - 10,
                        y - 8,
                        px
                        + largura_util
                        + 10,
                        y
                        + altura_linha
                        - 16
                    ],
                    radius=12,
                    fill=(
                        *cor,
                        24
                    )
                )

            desenho.text(
                (
                    px,
                    y
                ),
                Emocoes.label(
                    nome_ingles
                ),
                font=self.fontes.barra_label,
                fill=Cores.TEXTO_ESCURO
            )

            x_barra = (
                px
                + largura_label
            )

            Desenho.barra(
                desenho,
                x_barra,
                y + 3,
                largura_barra,
                altura_barra,
                valor,
                cor
            )

            texto_pct = (
                f"{valor:.0f}%"
            )

            caixa_pct = desenho.textbbox(
                (0, 0),
                texto_pct,
                font=self.fontes.barra_pct
            )

            x_pct = (
                px
                + largura_util
                - (
                    caixa_pct[2]
                    - caixa_pct[0]
                )
            )

            desenho.text(
                (
                    x_pct,
                    y
                ),
                texto_pct,
                font=self.fontes.barra_pct,
                fill=Cores.TEXTO_ESCURO
            )


# ============================================================
# OVERLAY DA CÂMERA
# ============================================================

class OverlayCamera:

    def desenhar_marcadores_rosto(
        self,
        frame,
        x,
        y,
        w,
        h,
        cor_bgr
    ):

        comprimento = int(
            min(w, h) * 0.22
        )

        espessura = 4

        cv2.line(
            frame,
            (x, y),
            (x + comprimento, y),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x, y),
            (x, y + comprimento),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x + w, y),
            (x + w - comprimento, y),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x + w, y),
            (x + w, y + comprimento),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x, y + h),
            (x + comprimento, y + h),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x, y + h),
            (x, y + h - comprimento),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x + w, y + h),
            (x + w - comprimento, y + h),
            cor_bgr,
            espessura
        )

        cv2.line(
            frame,
            (x + w, y + h),
            (x + w, y + h - comprimento),
            cor_bgr,
            espessura
        )

    def desenhar_chip_emocao(
        self,
        frame,
        x,
        y,
        texto,
        cor_bgr
    ):

        fonte = cv2.FONT_HERSHEY_SIMPLEX

        escala = 0.8

        espessura = 2

        (
            largura_texto,
            altura_texto
        ), _ = cv2.getTextSize(
            texto,
            fonte,
            escala,
            espessura
        )

        pad_x = 16
        pad_y = 10

        x1 = x

        y1 = max(
            y
            - altura_texto
            - pad_y * 2
            - 12,
            10
        )

        x2 = (
            x
            + largura_texto
            + pad_x * 2
        )

        y2 = (
            y1
            + altura_texto
            + pad_y * 2
        )

        Desenho.retangulo_arredondado_cv2(
            frame,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            12,
            cor_bgr
        )

        cv2.putText(
            frame,
            texto,
            (
                x1 + pad_x,
                y2 - pad_y - 4
            ),
            fonte,
            escala,
            Cores.bgr(
                Cores.BRANCO
            ),
            espessura,
            cv2.LINE_AA
        )

    def desenhar_status(
        self,
        frame,
        rosto_detectado
    ):

        pulso = (
            5
            + int(
                2
                * (
                    0.5
                    + 0.5
                    * math.sin(
                        time.time()
                        * 3
                    )
                )
            )
        )

        cv2.circle(
            frame,
            (
                34,
                34
            ),
            pulso,
            Cores.bgr(
                Cores.AO_VIVO
            ),
            -1
        )

        cv2.putText(
            frame,
            "AO VIVO",
            (
                52,
                42
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            Cores.bgr(
                Cores.BRANCO
            ),
            2,
            cv2.LINE_AA
        )

        largura_frame = (
            frame.shape[1]
        )

        if rosto_detectado:

            cor_status = (
                Cores.SUCESSO
            )

            texto_status = (
                "ROSTO DETECTADO"
            )

        else:

            cor_status = (
                Cores.NEUTRO_STATUS
            )

            texto_status = (
                "PROCURANDO ROSTO..."
            )

        (
            largura_texto,
            _
        ), _ = cv2.getTextSize(
            texto_status,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2
        )

        x_texto = (
            largura_frame
            - largura_texto
            - 46
        )

        cv2.circle(
            frame,
            (
                largura_frame - 30,
                34
            ),
            8,
            Cores.bgr(
                cor_status
            ),
            -1
        )

        cv2.putText(
            frame,
            texto_status,
            (
                x_texto,
                42
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            Cores.bgr(
                Cores.BRANCO
            ),
            2,
            cv2.LINE_AA
        )

    def aplicar(
        self,
        frame,
        rosto_detectado,
        resultado_atual
    ):

        if (
            resultado_atual is not None
            and rosto_detectado
        ):

            rx = resultado_atual["x"]
            ry = resultado_atual["y"]
            rw = resultado_atual["w"]
            rh = resultado_atual["h"]

            cor_atual = Cores.bgr(
                Emocoes.cor(
                    resultado_atual[
                        "emocao_ingles"
                    ]
                )
            )

            self.desenhar_marcadores_rosto(
                frame,
                rx,
                ry,
                rw,
                rh,
                cor_atual
            )

            texto = (
                f"{resultado_atual['emocao']} "
                f"{resultado_atual['confianca']:.0f}%"
            )

            self.desenhar_chip_emocao(
                frame,
                rx,
                ry,
                texto,
                cor_atual
            )

        self.desenhar_status(
            frame,
            rosto_detectado
        )

        return frame


# ============================================================
# RELATÓRIO
# ============================================================

class RelatorioSiren:

    NOME_JANELA = (
        "SIREN - Relatorio da sessao"
    )

    def gerar(
        self,
        dados_sessao
    ):

        if not dados_sessao:

            self._mostrar_aviso_sem_dados()

            return

        self._mostrar_grafico(
            dados_sessao
        )

    def fechar(self):

        try:

            cv2.destroyWindow(
                self.NOME_JANELA
            )

        except Exception:
            pass

    def _mostrar_aviso_sem_dados(self):

        aviso = np.full(
            (
                300,
                700,
                3
            ),
            255,
            dtype=np.uint8
        )

        cv2.putText(
            aviso,
            "Ainda nao ha dados suficientes nesta sessao.",
            (
                35,
                125
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (
                60,
                60,
                60
            ),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            aviso,
            "Posicione o rosto na camera e aguarde.",
            (
                35,
                165
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                110,
                110,
                110
            ),
            2,
            cv2.LINE_AA
        )

        cv2.namedWindow(
            self.NOME_JANELA,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.imshow(
            self.NOME_JANELA,
            aviso
        )

    def _mostrar_grafico(
        self,
        dados_sessao
    ):

        contagem = {}

        for (
            _,
            emocao_ingles,
            _
        ) in dados_sessao:

            contagem[
                emocao_ingles
            ] = (
                contagem.get(
                    emocao_ingles,
                    0
                )
                + 1
            )

        itens_ordenados = sorted(
            contagem.items(),
            key=lambda item: item[1],
            reverse=True
        )

        rotulos = [
            Emocoes.label(
                nome
            )
            for nome, _ in itens_ordenados
        ]

        valores = [
            valor
            for _, valor
            in itens_ordenados
        ]

        cores = [
            tuple(
                c / 255
                for c in Emocoes.cor(
                    nome
                )
            )
            for nome, _ in itens_ordenados
        ]

        total = len(
            dados_sessao
        )

        inicio = dados_sessao[0][0]

        fim = dados_sessao[-1][0]

        data_str = inicio.strftime(
            "%d/%m/%Y"
        )

        intervalo_str = (
            f"{inicio.strftime('%H:%M:%S')}"
            f" - "
            f"{fim.strftime('%H:%M:%S')}"
        )

        fig, ax = plt.subplots(
            figsize=(7.2, 6.2),
            dpi=100
        )

        fig.patch.set_facecolor(
            "white"
        )

        ax.pie(
            valores,
            labels=rotulos,
            colors=cores,
            autopct=lambda p: f"{p:.1f}%",
            startangle=90,
            pctdistance=0.75,
            textprops={
                "fontsize": 11,
                "color": "#28293D"
            },
            wedgeprops={
                "edgecolor": "white",
                "linewidth": 2
            }
        )

        ax.set_title(
            f"Picos de emoção — {data_str}\n"
            f"{intervalo_str} | "
            f"{total} análises registradas",
            fontsize=13,
            color="#28293D",
            pad=18
        )

        ax.axis("equal")

        fig.tight_layout()

        imagem = (
            self._figura_para_bgr(
                fig
            )
        )

        plt.close(fig)

        cv2.namedWindow(
            self.NOME_JANELA,
            cv2.WINDOW_AUTOSIZE
        )

        cv2.imshow(
            self.NOME_JANELA,
            imagem
        )

    @staticmethod
    def _figura_para_bgr(
        fig
    ):

        canvas = (
            FigureCanvasAgg(
                fig
            )
        )

        canvas.draw()

        buffer = np.asarray(
            canvas.buffer_rgba()
        )

        imagem_rgb = (
            buffer[:, :, :3]
        )

        return cv2.cvtColor(
            imagem_rgb,
            cv2.COLOR_RGB2BGR
        )


# ============================================================
# INTERFACE SIREN
# ============================================================

class InterfaceSiren:
    """
    Interface gráfica completa do SIREN.
    """

    NOME_JANELA = "SIREN"

    def __init__(self):

        self.fontes = Fontes()

        self.logo = LogoSiren()

        self.painel = (
            PainelEmocoes(
                self.fontes
            )
        )

        self.overlay_camera = (
            OverlayCamera()
        )

        self.relatorio = (
            RelatorioSiren()
        )

        self.relatorio_aberto = False

        self.botao_hover = False

        self._base_pil = (
            self._construir_base()
        )

        self._mascara_camera = (
            Desenho.mascara_arredondada(
                Layout.CAMERA_LARGURA,
                Layout.CAMERA_ALTURA,
                Layout.RAIO_CARTAO
            )
        )

        self._tempos_frame = []

        self._configurar_janela()

    # --------------------------------------------------------
    # JANELA
    # --------------------------------------------------------

    def _configurar_janela(self):

        cv2.namedWindow(
            self.NOME_JANELA,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            self.NOME_JANELA,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

    def definir_callback_mouse(
        self,
        callback
    ):

        cv2.setMouseCallback(
            self.NOME_JANELA,
            callback
        )

    def ponto_dentro_botao(
        self,
        x,
        y
    ):

        (
            x1,
            y1,
            x2,
            y2
        ) = Layout.BOTAO_RELATORIO_RECT

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )

    def alternar_relatorio(
        self,
        dados_sessao
    ):

        if self.relatorio_aberto:

            self.relatorio.fechar()

            self.relatorio_aberto = False

        else:

            self.relatorio.gerar(
                dados_sessao
            )

            self.relatorio_aberto = True

    # --------------------------------------------------------
    # BASE
    # --------------------------------------------------------

    def _construir_base(self):

        fundo_rgb = Desenho.gradiente(
            Layout.LARGURA_TELA,
            Layout.ALTURA_TELA,
            Cores.FUNDO_TOPO,
            Cores.FUNDO_BASE
        )

        canvas = Image.fromarray(
            fundo_rgb,
            "RGB"
        ).convert(
            "RGBA"
        )

        desenho = ImageDraw.Draw(
            canvas
        )

        Desenho.texto_centralizado(
            desenho,
            "SIREN",
            28,
            self.fontes.titulo,
            Cores.BRANCO
        )

        Desenho.texto_centralizado(
            desenho,
            (
                "Sistema de Inteligência "
                "Robótica para Regulação do Humor"
            ),
            104,
            self.fontes.subtitulo,
            Cores.TEXTO_CLARO_SUAVE
        )

        self.logo.colar(
            canvas,
            self.fontes
        )

        Desenho.sombra(
            canvas,
            Layout.CAMERA_X,
            Layout.CAMERA_Y,
            Layout.CAMERA_LARGURA,
            Layout.CAMERA_ALTURA,
            Layout.RAIO_CARTAO
        )

        Desenho.sombra(
            canvas,
            Layout.PAINEL_X,
            Layout.PAINEL_Y,
            Layout.PAINEL_LARGURA,
            Layout.PAINEL_ALTURA,
            Layout.RAIO_CARTAO
        )

        desenho.rounded_rectangle(
            [
                Layout.PAINEL_X,
                Layout.PAINEL_Y,
                Layout.PAINEL_X
                + Layout.PAINEL_LARGURA,
                Layout.PAINEL_Y
                + Layout.PAINEL_ALTURA
            ],
            radius=Layout.RAIO_CARTAO,
            fill=(
                255,
                255,
                255,
                255
            ),
            outline=(
                *Cores.CARTAO_BORDA,
                255
            ),
            width=2
        )

        desenho.rounded_rectangle(
            [
                Layout.MARGEM,
                Layout.RODAPE_Y,
                Layout.LARGURA_TELA
                - Layout.MARGEM,
                Layout.RODAPE_Y
                + Layout.RODAPE_ALTURA
            ],
            radius=16,
            fill=(
                255,
                255,
                255,
                22
            )
        )

        return canvas

    # --------------------------------------------------------
    # CÂMERA
    # --------------------------------------------------------

    def _colar_camera(
        self,
        tela_pil,
        frame_bgr
    ):

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        frame_pil = Image.fromarray(
            frame_rgb,
            "RGB"
        )

        tela_pil.paste(
            frame_pil,
            (
                Layout.CAMERA_X,
                Layout.CAMERA_Y
            ),
            self._mascara_camera
        )

    # --------------------------------------------------------
    # RODAPÉ
    # --------------------------------------------------------

    def _desenhar_rodape(
        self,
        tela_pil,
        fps,
        resultado
    ):

        desenho = ImageDraw.Draw(
            tela_pil
        )

        y_texto = (
            Layout.RODAPE_Y
            + (
                Layout.RODAPE_ALTURA
                - 18
            ) // 2
        )

        desenho.text(
            (
                Layout.MARGEM + 24,
                y_texto
            ),
            "Pressione Q para sair",
            font=self.fontes.rodape,
            fill=Cores.TEXTO_CLARO_SUAVE
        )

        if resultado is not None:

            texto_direita = (
                f"Processamento: "
                f"{resultado['tempo_processamento'] * 1000:.0f} ms"
                f"   |   "
                f"{fps:.0f} FPS"
            )

        else:

            texto_direita = (
                f"{fps:.0f} FPS"
            )

        caixa = desenho.textbbox(
            (0, 0),
            texto_direita,
            font=self.fontes.rodape
        )

        x_direita = (
            Layout.LARGURA_TELA
            - Layout.MARGEM
            - 24
            - (
                caixa[2]
                - caixa[0]
            )
        )

        desenho.text(
            (
                x_direita,
                y_texto
            ),
            texto_direita,
            font=self.fontes.rodape,
            fill=Cores.TEXTO_CLARO_SUAVE
        )

    # --------------------------------------------------------
    # BOTÃO
    # --------------------------------------------------------

    def _desenhar_botao_relatorio(
        self,
        tela_pil
    ):

        desenho = ImageDraw.Draw(
            tela_pil
        )

        (
            x1,
            y1,
            x2,
            y2
        ) = Layout.BOTAO_RELATORIO_RECT

        if self.relatorio_aberto:

            cor = (
                Cores.ACENTO_BOTAO_ATIVO
            )

        elif self.botao_hover:

            cor = (
                Cores.ACENTO_HOVER
            )

        else:

            cor = (
                Cores.ACENTO_BOTAO
            )

        desenho.rounded_rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            radius=16,
            fill=(
                *cor,
                255
            )
        )

        texto = (
            "OCULTAR RELATÓRIO"
            if self.relatorio_aberto
            else
            "GERAR RELATÓRIO"
        )

        caixa = desenho.textbbox(
            (0, 0),
            texto,
            font=self.fontes.botao
        )

        largura_texto = (
            caixa[2]
            - caixa[0]
        )

        altura_texto = (
            caixa[3]
            - caixa[1]
        )

        cx = (
            x1
            + (
                x2 - x1
                - largura_texto
            ) // 2
        )

        cy = (
            y1
            + (
                y2 - y1
                - altura_texto
            ) // 2
            - caixa[1]
        )

        desenho.text(
            (
                cx,
                cy
            ),
            texto,
            font=self.fontes.botao,
            fill=Cores.BRANCO
        )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    def _atualizar_fps(
        self,
        duracao_frame
    ):

        self._tempos_frame.append(
            duracao_frame
        )

        if len(
            self._tempos_frame
        ) > 30:

            self._tempos_frame.pop(
                0
            )

        media = (
            sum(
                self._tempos_frame
            )
            /
            len(
                self._tempos_frame
            )
        )

        if media <= 0:
            return 0

        return 1.0 / media

    # --------------------------------------------------------
    # RENDER
    # --------------------------------------------------------

    def renderizar(
        self,
        frame_camera_bgr,
        rosto_detectado,
        resultado_atual,
        duracao_frame
    ):

        frame_com_overlay = (
            self.overlay_camera.aplicar(
                frame_camera_bgr,
                rosto_detectado,
                resultado_atual
            )
        )

        frame_redimensionado = cv2.resize(
            frame_com_overlay,
            (
                Layout.CAMERA_LARGURA,
                Layout.CAMERA_ALTURA
            ),
            interpolation=cv2.INTER_AREA
        )

        fps = (
            self._atualizar_fps(
                duracao_frame
            )
        )

        tela_pil = (
            self._base_pil.copy()
        )

        self._colar_camera(
            tela_pil,
            frame_redimensionado
        )

        self.painel.desenhar(
            tela_pil,
            resultado_atual
        )

        self._desenhar_rodape(
            tela_pil,
            fps,
            resultado_atual
        )

        self._desenhar_botao_relatorio(
            tela_pil
        )

        return cv2.cvtColor(
            np.array(tela_pil),
            cv2.COLOR_RGBA2BGR
        )

    def mostrar(
        self,
        tela_final_bgr
    ):

        cv2.imshow(
            self.NOME_JANELA,
            tela_final_bgr
        )


# ============================================================
# TESTE DO FRONTEND
# ============================================================

if __name__ == "__main__":

    print(
        "Frontend SIREN carregado corretamente."
    )

    print(
        "InterfaceSiren:",
        InterfaceSiren
    )
