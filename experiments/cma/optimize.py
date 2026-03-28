"""CMA-ES optimization of SealBot pattern evaluation weights.

Optimizes the 364 free parameters (from 729 total, halved by player-swap
symmetry) using CMA-ES. Fitness is measured by win rate against the
baseline (hardcoded best/ weights) over a batch of games.

Usage:
    python optimize.py                     # defaults
    python optimize.py --games 30 --popsize 80
    python optimize.py --resume            # resume from checkpoint
"""

import argparse
import multiprocessing as mp
import os
import pickle
import sys
import time

import cma
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pkl")

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from symmetry import free_to_full, full_to_free, load_baseline, save_pattern_data_h


# ── Global config (set by main, read by workers) ───────────────────────────

_CFG = {}


def _init_worker(cfg):
    """Initializer for each pool worker -- import the C++ module once."""
    global _CFG
    _CFG = cfg
    # Import here so each worker process has its own module instance
    sys.path.insert(0, cfg["script_dir"])
    sys.path.insert(0, cfg["root_dir"])


def _evaluate_one(free_params):
    """Fitness function: play games, return negative win rate.

    CMA-ES minimizes, so better candidates have more negative fitness.
    """
    from game import HexGame, Player
    import cma_minimax_cpp

    num_games = _CFG["num_games"]
    time_limit = _CFG["time_limit"]

    full_params = free_to_full(free_params)

    score = 0.0
    for game_idx in range(num_games):
        swapped = game_idx % 2 == 1

        # Fresh bots each game (clean transposition tables etc.)
        candidate = cma_minimax_cpp.MinimaxBot(time_limit)
        candidate.load_patterns(full_params)
        baseline = cma_minimax_cpp.MinimaxBot(time_limit)

        if swapped:
            bot_a, bot_b = baseline, candidate
        else:
            bot_a, bot_b = candidate, baseline

        try:
            winner = _play_game(bot_a, bot_b, time_limit)
        except Exception:
            winner = Player.NONE

        # Score from candidate's perspective
        if swapped:
            if winner == Player.B:
                score += 1.0
            elif winner == Player.NONE:
                score += 0.5
        else:
            if winner == Player.A:
                score += 1.0
            elif winner == Player.NONE:
                score += 0.5

    win_rate = score / num_games
    return -win_rate  # negate: CMA-ES minimizes


def _play_game(bot_a, bot_b, time_limit, max_moves=200):
    """Play one game between two bot engines. Returns the winner."""
    from game import HexGame, Player

    game = HexGame(win_length=6)
    bots = {Player.A: bot_a, Player.B: bot_b}
    total = 0

    while not game.game_over and total < max_moves:
        player = game.current_player
        bot = bots[player]
        bot.time_limit = time_limit
        moves = bot.get_move(game)

        if not moves:
            return Player.B if player == Player.A else Player.A

        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                return Player.B if player == Player.A else Player.A
        total += len(moves)

    return game.winner


# ── Main optimization loop ─────────────────────────────────────────────────

def run(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load starting point
    baseline_full = load_baseline()
    x0 = full_to_free(baseline_full)
    print(f"Free parameters: {len(x0)} (from {len(baseline_full)} total)")
    print(f"Parameter stats: median |x|={np.median(np.abs(x0)):.0f}, "
          f"mean |x|={np.mean(np.abs(x0)):.0f}, max |x|={np.max(np.abs(x0)):.0f}")

    # CMA-ES options
    opts = cma.CMAOptions()
    opts["popsize"] = args.popsize
    opts["maxiter"] = args.max_gen
    opts["seed"] = args.seed
    opts["verb_disp"] = 1
    opts["verb_filenameprefix"] = os.path.join(OUTPUT_DIR, "outcma_")
    opts["verb_log"] = 1
    # Noise handling for stochastic fitness
    opts["noise_handling"] = True
    opts["tolfun"] = 1e-6

    # Resume or start fresh
    if args.resume and os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from {CHECKPOINT_PATH}")
        with open(CHECKPOINT_PATH, "rb") as f:
            state = pickle.load(f)
        es = state["es"]
        best_fitness = state["best_fitness"]
        best_free = state["best_free"]
        gen_offset = state["generation"]
        print(f"  Resuming at generation {gen_offset}, best fitness {best_fitness:.4f}")
    else:
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, opts)
        best_fitness = 0.0   # worst possible (win rate = 0)
        best_free = x0.copy()
        gen_offset = 0

    # Worker config
    cfg = {
        "num_games": args.games,
        "time_limit": args.time_limit,
        "script_dir": SCRIPT_DIR,
        "root_dir": ROOT_DIR,
    }

    num_workers = args.workers or mp.cpu_count()
    print(f"\nCMA-ES: sigma0={args.sigma0}, popsize={args.popsize}, "
          f"games/eval={args.games}, time_limit={args.time_limit}s")
    print(f"Workers: {num_workers}, max generations: {args.max_gen}")
    print(f"Output: {OUTPUT_DIR}\n")

    gen = gen_offset
    t_start = time.time()

    try:
        while not es.stop():
            gen += 1
            t_gen = time.time()

            # Ask for candidate solutions
            solutions = es.ask()

            # Evaluate in parallel
            with mp.Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
                fitnesses = pool.map(_evaluate_one, solutions)

            es.tell(solutions, fitnesses)
            es.disp()

            # Track best
            gen_best_idx = int(np.argmin(fitnesses))
            gen_best_fit = fitnesses[gen_best_idx]
            if gen_best_fit < best_fitness:
                best_fitness = gen_best_fit
                best_free = np.array(solutions[gen_best_idx])

            elapsed_gen = time.time() - t_gen
            elapsed_total = time.time() - t_start
            print(f"  Gen {gen}: best_wr={-best_fitness:.1%}, "
                  f"gen_best_wr={-gen_best_fit:.1%}, "
                  f"mean_wr={-np.mean(fitnesses):.1%}, "
                  f"sigma={es.sigma:.1f}, "
                  f"gen_time={elapsed_gen:.0f}s, "
                  f"total={elapsed_total/60:.0f}min")

            # Checkpoint every generation
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump({
                    "es": es,
                    "best_fitness": best_fitness,
                    "best_free": best_free,
                    "generation": gen,
                }, f)

            # Save best pattern_data.h every 10 generations
            if gen % 10 == 0:
                best_full = free_to_full(best_free)
                out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
                save_pattern_data_h(best_full, out_path)
                print(f"  Saved checkpoint: {out_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Final save
    best_full = free_to_full(best_free)
    out_path = os.path.join(OUTPUT_DIR, "best_pattern_data.h")
    save_pattern_data_h(best_full, out_path)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  CMA-ES finished after {gen - gen_offset} generations ({elapsed/60:.0f} min)")
    print(f"  Best win rate vs baseline: {-best_fitness:.1%}")
    print(f"  Output: {out_path}")
    print(f"\n  To use these weights:")
    print(f"    cp {out_path} ../../current/pattern_data.h")
    print(f"    cd ../.. && make rebuild")
    print(f"    python evaluate.py -n 100 -t 0.1")
    print(f"{'='*50}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMA-ES optimization of SealBot pattern weights")

    parser.add_argument("--games", type=int, default=20,
                        help="Games per fitness evaluation (default: 20)")
    parser.add_argument("--time-limit", type=float, default=0.02,
                        help="Seconds per move during evaluation (default: 0.02)")
    parser.add_argument("--popsize", type=int, default=50,
                        help="CMA-ES population size (default: 50)")
    parser.add_argument("--sigma0", type=float, default=50.0,
                        help="CMA-ES initial step size (default: 50.0)")
    parser.add_argument("--max-gen", type=int, default=500,
                        help="Maximum generations (default: 500)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: cpu_count)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    run(parser.parse_args())
