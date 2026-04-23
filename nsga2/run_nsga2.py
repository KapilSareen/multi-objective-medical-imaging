"""
Phase 2: NSGA-II Evolution
Multi-objective optimization of ensemble weights
"""

import sys
import argparse
import pickle
from pathlib import Path
import numpy as np
import json
from deap import base, creator, tools, algorithms
import multiprocessing as mp
from tqdm import tqdm
import time

# Import objectives
sys.path.insert(0, str(Path(__file__).parent))
from objectives import evaluate_ensemble, batch_evaluate


def create_individual(n_models):
    """Create random weight vector"""
    weights = np.random.dirichlet(np.ones(n_models))  # Sum to 1
    return weights.tolist()


def mutate_weights(individual, eta=20, indpb=0.2):
    """Polynomial mutation for weights"""
    individual = np.array(individual)
    
    for i in range(len(individual)):
        if np.random.rand() < indpb:
            # Polynomial mutation
            delta = np.random.rand()
            if delta < 0.5:
                delta_q = (2 * delta) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - delta)) ** (1 / (eta + 1))
            
            individual[i] = individual[i] + delta_q
    
    # Ensure non-negative and normalize
    individual = np.abs(individual)
    individual = individual / individual.sum()
    
    return individual.tolist(),


def crossover_weights(ind1, ind2, eta=20):
    """SBX crossover for weights"""
    ind1 = np.array(ind1)
    ind2 = np.array(ind2)
    
    for i in range(len(ind1)):
        if np.random.rand() < 0.5:
            u = np.random.rand()
            
            if u <= 0.5:
                beta = (2 * u) ** (1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            
            c1 = 0.5 * ((1 + beta) * ind1[i] + (1 - beta) * ind2[i])
            c2 = 0.5 * ((1 - beta) * ind1[i] + (1 + beta) * ind2[i])
            
            ind1[i] = c1
            ind2[i] = c2
    
    # Normalize
    ind1 = np.abs(ind1) / np.abs(ind1).sum()
    ind2 = np.abs(ind2) / np.abs(ind2).sum()
    
    return ind1.tolist(), ind2.tolist()


_SHARED = {}


def _worker_init(P_cache, y_true, demographics):
    """Called once per worker process to load shared data into global"""
    _SHARED['P_cache'] = P_cache
    _SHARED['y_true'] = y_true
    _SHARED['demographics'] = demographics


def evaluate_wrapper(individual):
    """Worker-side evaluation using process-local shared data"""
    return evaluate_ensemble(individual, _SHARED['P_cache'], _SHARED['y_true'], _SHARED['demographics'])


def save_checkpoint(generation, population, logbook, checkpoint_path):
    """Save evolution checkpoint"""
    checkpoint = {
        'generation': generation,
        'population': population,
        'logbook': logbook,
    }
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(checkpoint_path):
    """Load evolution checkpoint"""
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint['generation'], checkpoint['population'], checkpoint['logbook']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pop_size', type=int, default=100)
    parser.add_argument('--n_gen', type=int, default=100)
    parser.add_argument('--n_workers', type=int, default=40)
    parser.add_argument('--cache_dir', type=str, default='data/cache')
    parser.add_argument('--output_dir', type=str, default='results/nsga2')
    parser.add_argument('--checkpoint_interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    print("="*80)
    print("PHASE 2: NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
    print("="*80)
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / args.cache_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / "nsga2_checkpoint.pkl"
    
    # Load cached predictions
    print(f"\n📊 Loading cached predictions...")
    P_cache = np.load(cache_dir / "P_cache.npy")
    y_true = np.load(cache_dir / "y_true.npy")
    demographics = np.load(cache_dir / "demographics.npy", allow_pickle=True)
    # Ensure string dtype (not bytes)
    demographics = demographics.astype(str)
    
    n_samples, n_models = P_cache.shape
    print(f"   Samples: {n_samples:,}")
    print(f"   Models: {n_models}")
    
    # Setup DEAP
    # selTournamentDCD requires pop_size divisible by 4
    if args.pop_size % 4 != 0:
        args.pop_size += (4 - args.pop_size % 4)

    print(f"\n🧬 Setting up NSGA-II...")
    print(f"   Population size: {args.pop_size}")
    print(f"   Generations: {args.n_gen}")
    print(f"   Workers: {args.n_workers}")
    
    # Fitness: minimize all three (-AUC, ACE, AUC_gap)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    
    # Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: create_individual(n_models))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Genetic operators
    toolbox.register("mate", crossover_weights)
    toolbox.register("mutate", mutate_weights)
    toolbox.register("select", tools.selNSGA2)
    
    # Evaluation
    toolbox.register("evaluate", evaluate_wrapper)

    # Parallel evaluation - each worker gets data loaded once via initializer
    pool = mp.Pool(
        args.n_workers,
        initializer=_worker_init,
        initargs=(P_cache, y_true, demographics)
    )
    toolbox.register("map", pool.map)

    # Initialize or resume
    start_gen = 0
    logbook = tools.Logbook()
    
    if args.resume and checkpoint_path.exists():
        print(f"\n♻️  Resuming from checkpoint...")
        start_gen, population, logbook = load_checkpoint(checkpoint_path)
        print(f"   Starting from generation {start_gen + 1}")
    else:
        print(f"\n🎲 Initializing population...")
        population = toolbox.population(n=args.pop_size)
        # Evaluate and assign crowding_dist so selTournamentDCD works from gen 0
        fitnesses = list(toolbox.map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        population[:] = toolbox.select(population, args.pop_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    # Evolution
    print(f"\n🚀 Starting evolution...")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for gen in range(start_gen, args.n_gen):
        gen_start = time.time()
        
        print(f"\n📊 Generation {gen + 1}/{args.n_gen}")
        
        # Generate offspring via tournament selection
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.rand() < 0.9:
                new1, new2 = toolbox.mate(child1[:], child2[:])
                child1[:] = new1
                child2[:] = new2
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.rand() < 0.2:
                new_mut, = toolbox.mutate(mutant[:])
                mutant[:] = new_mut
                del mutant.fitness.values

        # Evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

        # NSGA-II selection: select from combined population + offspring
        population[:] = toolbox.select(population + offspring, args.pop_size)
        
        # Log statistics
        record = stats.compile(population)
        logbook.record(gen=gen + 1, **record)
        
        gen_time = time.time() - gen_start
        elapsed = time.time() - start_time
        
        print(f"   Min: {record['min']}")
        print(f"   Avg: {record['avg']}")
        print(f"   Max: {record['max']}")
        print(f"   Time: {gen_time:.1f}s (Total: {elapsed/60:.1f}m)")
        
        # Save checkpoint
        if (gen + 1) % args.checkpoint_interval == 0:
            save_checkpoint(gen + 1, population, logbook, checkpoint_path)
            print(f"   💾 Checkpoint saved")
    
    pool.close()
    pool.join()

    # Extract Pareto front
    print(f"\n{'='*80}")
    print("EXTRACTING PARETO FRONT")
    print(f"{'='*80}")
    
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    
    print(f"   Pareto front size: {len(pareto_front)}")
    
    # Save results
    pareto_weights = [ind[:] for ind in pareto_front]
    pareto_fitness = [ind.fitness.values for ind in pareto_front]
    
    np.save(output_dir / "pareto_weights.npy", np.array(pareto_weights))
    np.save(output_dir / "pareto_fitness.npy", np.array(pareto_fitness))
    
    # Save logbook
    with open(output_dir / "evolution_log.pkl", 'wb') as f:
        pickle.dump(logbook, f)
    
    # Save summary
    last = logbook[-1]
    summary = {
        'population_size': args.pop_size,
        'generations': args.n_gen,
        'pareto_size': len(pareto_front),
        'final_stats': {
            'min': np.array(last['min']).tolist(),
            'avg': np.array(last['avg']).tolist(),
        },
        'total_time_minutes': (time.time() - start_time) / 60
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved:")
    print(f"   {output_dir / 'pareto_weights.npy'}")
    print(f"   {output_dir / 'pareto_fitness.npy'}")
    print(f"   {output_dir / 'evolution_log.pkl'}")
    print(f"   {output_dir / 'summary.json'}")
    
    print(f"\n{'='*80}")
    print("✅ NSGA-II COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
