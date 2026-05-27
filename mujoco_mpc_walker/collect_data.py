"""
collect_data.py
---------------
Coleta dados do professor MPC em tempo real com visualizacao.

Estrategia:
  1. Lanca o mjpc.exe (com GUI completa) para voce ver o MPC em acao
  2. Em PARALELO, roda o generate_dataset.exe (headless) que realmente
     coleta os pares (estado, acao) do MPC e salva em dataset.csv
  3. Monitora o progresso e encerra automaticamente

Fluxo completo:
  python collect_data.py          -> coleta e fecha
  python collect_data.py --steps 30000 -> mais dados
  python train.py                 -> treina o student com os dados
  python play.py                  -> visualiza o student treinado
"""

import subprocess
import sys
import os
import time
import argparse
import threading

# ─────────────────────────────────────────────────
# Caminhos
# ─────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR       = os.path.join(SCRIPT_DIR, "build")
MJPC_EXE        = os.path.join(BUILD_DIR, "bin", "mjpc.exe")
GEN_EXE         = os.path.join(BUILD_DIR, "mjpc_dataset_tool.exe")
DATASET_CSV     = os.path.join(BUILD_DIR, "dataset.csv")


# ─────────────────────────────────────────────────
# Utilitarios
# ─────────────────────────────────────────────────
def check_executables():
    missing = []
    if not os.path.exists(MJPC_EXE):
        missing.append(MJPC_EXE)
    if not os.path.exists(GEN_EXE):
        missing.append(GEN_EXE)
    if missing:
        print("[ERRO] Os seguintes executaveis nao foram encontrados:")
        for m in missing:
            print(f"       {m}")
        print()
        print("Execute o build primeiro:")
        print("  .\\build.bat")
        sys.exit(1)


def count_dataset_lines():
    """Conta quantas linhas validas existem no CSV (excluindo header)."""
    if not os.path.exists(DATASET_CSV):
        return 0
    try:
        with open(DATASET_CSV, "r") as f:
            return max(0, sum(1 for _ in f) - 1)  # -1 para o header
    except Exception:
        return 0


def stream_output(proc, prefix="[GEN]"):
    """Lê a saída do processo em uma thread separada e imprime com prefixo."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"\r{prefix} {text}                    ", flush=True)


# ─────────────────────────────────────────────────
# Processo principal
# ─────────────────────────────────────────────────
def launch_gui():
    """Lanca o mjpc.exe com interface grafica completa."""
    print(f"[GUI] Iniciando mjpc.exe com visualizacao...")
    proc = subprocess.Popen(
        [MJPC_EXE, "--task=Walker Ragdoll"],
        cwd=BUILD_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )
    print(f"[GUI] mjpc.exe iniciado (PID {proc.pid})")
    print(f"[GUI] Voce pode ver o MPC em acao na janela que abriu!")
    print(f"[GUI] (Pode fechar a janela do mjpc quando a coleta terminar)")
    print()
    return proc


def launch_collector(target_steps: int):
    """
    Lanca o generate_dataset.exe que coleta dados headless.
    Captura a saida para monitorar o progresso.
    """
    print(f"[COLETA] Iniciando generate_dataset.exe ({target_steps} passos)...")
    
    # Passa o numero de passos via variavel de ambiente (ou argumento futuro)
    env = os.environ.copy()
    env["TARGET_STEPS"] = str(target_steps)
    
    # Adiciona build/bin ao PATH para que o windows ache a mujoco.dll
    bin_dir = os.path.join(BUILD_DIR, "bin")
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    
    proc = subprocess.Popen(
        [GEN_EXE],
        cwd=BUILD_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    return proc


def monitor_progress(gen_proc, target_steps: int):
    """Monitora o progresso da coleta e imprime em tempo real."""
    start_time = time.time()
    last_count = 0

    print(f"[COLETA] Aguardando dados em {DATASET_CSV}...")
    print()

    # Lê a saída do processo em thread separada
    output_thread = threading.Thread(
        target=stream_output, args=(gen_proc, "[GEN]"), daemon=True
    )
    output_thread.start()

    while gen_proc.poll() is None:
        time.sleep(2)
        current_count = count_dataset_lines()
        elapsed = time.time() - start_time
        rate = current_count / max(elapsed, 1)
        eta = (target_steps - current_count) / max(rate, 0.1)

        bar_len = 30
        filled = int(bar_len * current_count / max(target_steps, 1))
        bar = "=" * filled + "-" * (bar_len - filled)

        print(
            f"\r[PROGRESSO] [{bar}] {current_count:6d}/{target_steps}"
            f" | {rate:.0f} steps/s | ETA: {int(eta)}s   ",
            end="", flush=True
        )
        last_count = current_count

    print()
    return gen_proc.returncode


# ─────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Coleta dados do professor MPC com visualizacao"
    )
    parser.add_argument(
        "--steps", type=int, default=100_000,
        help="Numero de passos validos a coletar (padrao: 100000)"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Nao lanca a janela do mjpc.exe (apenas coleta headless)"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  Coleta de Dados do Professor MPC")
    print("=" * 60)
    print()
    print(f"  Alvo: {args.steps:,} passos validos")
    print(f"  Saida: {DATASET_CSV}")
    print()

    check_executables()

    # Lanca a GUI para visualizacao (opcional)
    gui_proc = None
    if not args.no_gui:
        gui_proc = launch_gui()
        print("[INFO] Aguardando 2s para a GUI carregar...")
        time.sleep(2)

    # Lanca o coletor headless em paralelo
    gen_proc = launch_collector(args.steps)

    try:
        returncode = monitor_progress(gen_proc, args.steps)
    except KeyboardInterrupt:
        print("\n[AVISO] Interrompido pelo usuario.")
        gen_proc.terminate()
        returncode = 1
    finally:
        if gui_proc and gui_proc.poll() is None:
            print("[INFO] Encerrando a GUI do mjpc...")
            gui_proc.terminate()

    final_count = count_dataset_lines()

    if final_count >= args.steps:
        print()
        print("=" * 60)
        print(f"  Coleta concluida com sucesso!")
        print(f"  Total de amostras: {final_count:,}")
        print(f"  Arquivo: {DATASET_CSV}")
        print()
        print("  Proximo passo: python train.py")
        print("=" * 60)
    elif final_count > 0:
        print()
        print(f"[AVISO] Coleta parcial: {final_count:,} de {args.steps:,} passos")
        print(f"  Arquivo: {DATASET_CSV}")
        print(f"  Voce pode treinar com os dados existentes: python train.py")
        print(f"  Ou coletar mais: python collect_data.py --steps {args.steps}")
    else:
        print(f"\n[ERRO] Nenhum dado foi coletado. Exit code: {returncode}")
        print("       Verifique se o build foi feito corretamente com: .\\build.bat")
        sys.exit(1)


if __name__ == "__main__":
    main()
