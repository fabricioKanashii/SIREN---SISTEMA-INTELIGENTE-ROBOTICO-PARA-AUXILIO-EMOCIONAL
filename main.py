import os
import sys
import time
import traceback


# ============================================================
# CAMINHO DO PROJETO
# ============================================================

PASTA_PROJETO = os.path.dirname(
    os.path.abspath(__file__)
)

if PASTA_PROJETO not in sys.path:

    sys.path.insert(
        0,
        PASTA_PROJETO
    )


# ============================================================
# IMPORTAÇÕES
# ============================================================

try:

    import cv2

except Exception:

    print(
        "[ERRO] OpenCV não está instalado."
    )

    traceback.print_exc()

    sys.exit(1)


try:

    from config import (
        ConfiguracaoGeral
    )

except Exception:

    print(
        "[ERRO] Falha ao carregar config.py"
    )

    traceback.print_exc()

    sys.exit(1)


try:

    from backend import (
        MotorSiren
    )

except Exception:

    print(
        "[ERRO] Falha ao carregar backend.py"
    )

    traceback.print_exc()

    sys.exit(1)


try:

    from frontend import (
        InterfaceSiren
    )

except Exception:

    print(
        "[ERRO] Falha ao carregar frontend.py"
    )

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# APLICAÇÃO
# ============================================================

class AplicacaoSiren:
    """Aplicação principal do SIREN."""

    def __init__(self):

        print(
            "[SIREN] Criando motor..."
        )

        self.motor = (
            MotorSiren()
        )

        print(
            "[SIREN] Criando interface..."
        )

        self.interface = (
            InterfaceSiren()
        )

        self.interface.definir_callback_mouse(
            self._ao_clicar_mouse
        )

        self._rodando = False

    # ========================================================
    # MOUSE
    # ========================================================

    def _ao_clicar_mouse(
        self,
        evento,
        x,
        y,
        flags,
        param
    ):

        try:

            if evento == cv2.EVENT_MOUSEMOVE:

                self.interface.botao_hover = (
                    self.interface
                    .ponto_dentro_botao(
                        x,
                        y
                    )
                )

                return

            if evento != cv2.EVENT_LBUTTONDOWN:

                return

            if not self.interface.ponto_dentro_botao(
                x,
                y
            ):

                return

            print(
                "[SIREN] Gerando relatório..."
            )

            dados = (
                self.motor
                .obter_dados_sessao()
            )

            self.interface.alternar_relatorio(
                dados
            )

        except Exception:

            print(
                "[ERRO] Falha no botão de relatório."
            )

            traceback.print_exc()

    # ========================================================
    # INFORMAÇÕES
    # ========================================================

    def _imprimir_cabecalho(self):

        print()
        print("=" * 65)
        print("                           SIREN")
        print("=" * 65)

        print(
            "Sistema de Inteligência Robótica "
            "para Regulação do Humor"
        )

        print("=" * 65)

        print(
            f"Câmera: índice "
            f"{ConfiguracaoGeral.CAMERA_INDEX}"
        )

        print(
            "DeepFace: CPU"
        )

        print(
            f"Entrada: "
            f"{ConfiguracaoGeral.TAMANHO_DEEPFACE[0]}"
            f"x"
            f"{ConfiguracaoGeral.TAMANHO_DEEPFACE[1]}"
        )

        print(
            f"Intervalo: "
            f"{ConfiguracaoGeral.INTERVALO_ANALISE:.2f}s"
        )

        print(
            f"Suavização: "
            f"{ConfiguracaoGeral.HISTORICO_TAMANHO} análises"
        )

        print()

        if self.interface.logo.imagem is None:

            print(
                "[AVISO] Logo não encontrada."
            )

            print(
                "Opcional: coloque sua logo em:"
            )

            print(
                ConfiguracaoGeral.CAMINHO_LOGO_BASE
                + ".png"
            )

        print()

        print(
            "CONTROLES"
        )

        print(
            "Q = sair"
        )

        print(
            "Botão inferior direito = relatório"
        )

        print("=" * 65)
        print()

    # ========================================================
    # EXECUÇÃO
    # ========================================================

    def executar(self):

        self._imprimir_cabecalho()

        try:

            self.motor.iniciar()

        except Exception:

            print(
                "[ERRO] Não foi possível iniciar o motor."
            )

            traceback.print_exc()

            self._encerrar()

            return

        self._rodando = True

        print(
            "[SIREN] Sistema iniciado."
        )

        print(
            "[SIREN] Procurando rosto..."
        )

        print()

        try:

            while self._rodando:

                inicio_frame = (
                    time.time()
                )

                # --------------------------------------------
                # CÂMERA
                # --------------------------------------------

                sucesso, frame = (
                    self.motor.camera
                    .ler_frame()
                )

                if not sucesso:

                    print(
                        "[AVISO] Falha ao capturar frame."
                    )

                    time.sleep(
                        0.05
                    )

                    continue

                # --------------------------------------------
                # DETECÇÃO
                # --------------------------------------------

                rosto_detectado, _ = (
                    self.motor
                    .processar_frame(
                        frame
                    )
                )

                # --------------------------------------------
                # RESULTADO
                # --------------------------------------------

                resultado = (
                    self.motor
                    .obter_resultado_ia()
                )

                # --------------------------------------------
                # TEMPO
                # --------------------------------------------

                duracao_frame = (
                    time.time()
                    - inicio_frame
                )

                # --------------------------------------------
                # INTERFACE
                # --------------------------------------------

                tela = (
                    self.interface
                    .renderizar(
                        frame,
                        rosto_detectado,
                        resultado,
                        duracao_frame
                    )
                )

                self.interface.mostrar(
                    tela
                )

                # --------------------------------------------
                # TECLADO
                # --------------------------------------------

                tecla = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if tecla in (
                    ord("q"),
                    ord("Q")
                ):

                    break

        except KeyboardInterrupt:

            print(
                "\n[SIREN] Interrupção pelo usuário."
            )

        except Exception:

            print(
                "\n[ERRO] Erro durante execução:"
            )

            traceback.print_exc()

        finally:

            self._encerrar()

    # ========================================================
    # ENCERRAMENTO
    # ========================================================

    def _encerrar(self):

        if not self._rodando:
            return

        self._rodando = False

        print()
        print(
            "[SIREN] Encerrando..."
        )

        try:

            self.motor.encerrar()

        except Exception:

            print(
                "[AVISO] Erro ao encerrar motor."
            )

            traceback.print_exc()

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

        print(
            "[SIREN] Sistema encerrado."
        )


# ============================================================
# PONTO DE ENTRADA
# ============================================================

def main():

    print()
    print(
        "Inicializando SIREN..."
    )
    print()

    try:

        app = AplicacaoSiren()

        app.executar()

    except Exception:

        print()
        print("=" * 65)
        print(
            "[ERRO FATAL] SIREN não conseguiu iniciar."
        )
        print("=" * 65)

        traceback.print_exc()

        print("=" * 65)


# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":

    main()
