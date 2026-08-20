import os
from PIL import ImageFont


# ============================================================
# CAMINHO DO PROJETO
# ============================================================

PASTA_PROJETO = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================

class ConfiguracaoGeral:
    """Parâmetros gerais do sistema SIREN."""

    CAMERA_INDEX = 0

    # Intervalo mínimo entre análises do DeepFace
    INTERVALO_ANALISE = 0.35

    # Tamanho da imagem enviada para o DeepFace
    TAMANHO_DEEPFACE = (160, 160)

    # Quantidade de análises utilizadas na suavização
    HISTORICO_TAMANHO = 3

    # Configuração da câmera
    CAMERA_LARGURA = 1920
    CAMERA_ALTURA = 1080

    # Caminho da logo
    CAMINHO_LOGO_BASE = os.path.join(
        PASTA_PROJETO,
        "assets/imgs",
        "logo"
    )

    EXTENSOES_LOGO = [
        ".png",
        ".jpg",
        ".jpeg",
        ".webp"
    ]


# ============================================================
# LAYOUT
# ============================================================

class Layout:
    """Dimensões e posicionamento dos elementos da interface."""

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

    PAINEL_X = (
        CAMERA_X
        + CAMERA_LARGURA
        + GAP
    )

    PAINEL_Y = CONTEUDO_Y

    PAINEL_LARGURA = (
        LARGURA_TELA
        - MARGEM
        - PAINEL_X
    )

    PAINEL_ALTURA = ALTURA_CONTEUDO

    RODAPE_Y = (
        CONTEUDO_Y
        + ALTURA_CONTEUDO
        + GAP
    )

    RODAPE_ALTURA = (
        ALTURA_TELA
        - RODAPE_Y
        - 30
    )

    RAIO_CARTAO = 26

    PADDING_CARTAO = 30

    # --------------------------------------------------------
    # LOGO
    # --------------------------------------------------------

    LOGO_LARGURA_MAX = 150
    LOGO_ALTURA_MAX = 86
    LOGO_MARGEM_TOPO = 22

    LOGO_X2 = LARGURA_TELA - MARGEM

    LOGO_X1 = (
        LOGO_X2
        - LOGO_LARGURA_MAX
    )

    LOGO_Y1 = LOGO_MARGEM_TOPO

    LOGO_Y2 = (
        LOGO_Y1
        + LOGO_ALTURA_MAX
    )

    # --------------------------------------------------------
    # BOTÃO RELATÓRIO
    # --------------------------------------------------------

    BOTAO_RELATORIO_LARGURA = 300
    BOTAO_RELATORIO_ALTURA = 50

    BOTAO_RELATORIO_X = (
        LARGURA_TELA
        - MARGEM
        - BOTAO_RELATORIO_LARGURA
    )

    BOTAO_RELATORIO_Y = (
        RODAPE_Y
        + (
            RODAPE_ALTURA
            - BOTAO_RELATORIO_ALTURA
        ) // 2
    )

    BOTAO_RELATORIO_RECT = (
        BOTAO_RELATORIO_X,
        BOTAO_RELATORIO_Y,
        BOTAO_RELATORIO_X
        + BOTAO_RELATORIO_LARGURA,
        BOTAO_RELATORIO_Y
        + BOTAO_RELATORIO_ALTURA,
    )


# ============================================================
# CORES
# ============================================================

class Cores:
    """Paleta de cores do SIREN."""

    FUNDO_TOPO = (
        32,
        36,
        74
    )

    FUNDO_BASE = (
        72,
        78,
        138
    )

    BRANCO = (
        255,
        255,
        255
    )

    TEXTO_CLARO = (
        240,
        242,
        250
    )

    TEXTO_CLARO_SUAVE = (
        196,
        201,
        224
    )

    TEXTO_ESCURO = (
        40,
        43,
        61
    )

    TEXTO_MUTED = (
        140,
        145,
        168
    )

    CARTAO_BORDA = (
        228,
        230,
        240
    )

    TRILHA_BARRA = (
        232,
        234,
        242
    )

    SUCESSO = (
        76,
        217,
        140
    )

    NEUTRO_STATUS = (
        170,
        174,
        190
    )

    AO_VIVO = (
        235,
        70,
        70
    )

    ACENTO_BOTAO = (
        108,
        99,
        255
    )

    ACENTO_BOTAO_ATIVO = (
        86,
        78,
        214
    )

    ACENTO_HOVER = (
        125,
        116,
        255
    )

    @staticmethod
    def bgr(cor):
        """Converte RGB para BGR."""

        return (
            cor[2],
            cor[1],
            cor[0]
        )


# ============================================================
# EMOÇÕES
# ============================================================

class Emocoes:
    """Mapeamento das emoções do DeepFace."""

    MAPA = {

        "happy": {
            "label": "ALEGRIA",
            "cor": (
                255,
                196,
                60
            )
        },

        "surprise": {
            "label": "SURPRESA",
            "cor": (
                247,
                150,
                66
            )
        },

        "neutral": {
            "label": "NEUTRO",
            "cor": (
                156,
                162,
                180
            )
        },

        "sad": {
            "label": "TRISTEZA",
            "cor": (
                86,
                148,
                227
            )
        },

        "fear": {
            "label": "MEDO",
            "cor": (
                159,
                108,
                217
            )
        },

        "disgust": {
            "label": "NOJO",
            "cor": (
                108,
                191,
                130
            )
        },

        "angry": {
            "label": "RAIVA",
            "cor": (
                232,
                77,
                77
            )
        },
    }

    @classmethod
    def cor(cls, nome_ingles):

        return cls.MAPA.get(
            nome_ingles,
            {}
        ).get(
            "cor",
            Cores.NEUTRO_STATUS
        )

    @classmethod
    def label(cls, nome_ingles):

        return cls.MAPA.get(
            nome_ingles,
            {}
        ).get(
            "label",
            str(nome_ingles).upper()
        )


# ============================================================
# FONTES
# ============================================================

class Fontes:
    """Carregamento centralizado das fontes."""

    _CANDIDATOS_TITULO = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]

    _CANDIDATOS_REGULAR = [
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]

    _CANDIDATOS_SEMIBOLD = [
        "C:/Windows/Fonts/segoeuisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]

    def __init__(self):

        self.titulo = self._carregar(
            self._CANDIDATOS_TITULO,
            62
        )

        self.subtitulo = self._carregar(
            self._CANDIDATOS_REGULAR,
            22
        )

        self.rotulo_pequeno = self._carregar(
            self._CANDIDATOS_SEMIBOLD,
            16
        )

        self.emocao_grande = self._carregar(
            self._CANDIDATOS_SEMIBOLD,
            46
        )

        self.confianca = self._carregar(
            self._CANDIDATOS_REGULAR,
            20
        )

        self.barra_label = self._carregar(
            self._CANDIDATOS_REGULAR,
            18
        )

        self.barra_pct = self._carregar(
            self._CANDIDATOS_SEMIBOLD,
            18
        )

        self.rodape = self._carregar(
            self._CANDIDATOS_REGULAR,
            16
        )

        self.botao = self._carregar(
            self._CANDIDATOS_SEMIBOLD,
            18
        )

        self.logo_placeholder = self._carregar(
            self._CANDIDATOS_REGULAR,
            13
        )

    @staticmethod
    def _carregar(candidatos, tamanho):

        for caminho in candidatos:

            try:
                return ImageFont.truetype(
                    caminho,
                    tamanho
                )

            except Exception:
                continue

        return ImageFont.load_default()
