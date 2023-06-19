import argparse
import os
from utils import *
from pymoo.problems import get_problem
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My script")
    parser.add_argument("--problem1", type=int, help="First problem")
    parser.add_argument("--problem2", type=int, help="Second problem")
    parser.add_argument("--instance1", type=int, help="First instance")
    parser.add_argument("--instance2", type=int, help="Second instance")
    parser.add_argument("--run", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--alphas", type=int)
    parser.add_argument("--n_eval", type=int)
    args = parser.parse_args()
    
    save_dir = "runs"
    create_directory_if_not_exist(save_dir)
    file = f'{save_dir}/p1-{args.problem1}-p2-{args.problem2}-i1-{args.instance1}-i2-{args.instance2}-run-{args.run}.csv'
    
    if os.path.isfile(file) == False:
        problem1 = get_problem(f"bbob-f{args.problem1}-{args.instance1}", n_var=args.dim)
        problem2 = get_problem(f"bbob-f{args.problem2}-{args.instance2}", n_var=args.dim)

        dfs = []
        for alpha in tqdm(np.linspace(0, 1, num=args.alphas)):
            mix = CombinedProblem(problem1, problem2, alpha)
            data = run_algorithms(mix, n_runs=1, n_eval=args.n_eval)
            pdf = pd.DataFrame(data)
            pdf['problem1'] = args.problem1
            pdf['problem2'] = args.problem2
            pdf['instance1'] = args.instance1
            pdf['instance2'] = args.instance2
            pdf['alpha'] = alpha
            pdf['dim'] = args.dim
            pdf['algorithm_run'] = args.run
            pdf['optimum'] = mix.evaluate(mix.pareto_set())
            dfs.append(pdf)

        dfs = pd.concat(dfs)
        dfs.to_csv(file, index=False)
    else:
        print('Skip file', file)

