import argparse
from utils import *
from pymoo.problems import get_problem
from tqdm import tqdm
import pandas as pd
import os
from pflacco.classical_ela_features import *

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
    
    save_dir = "ela"
    create_directory_if_not_exist(save_dir)
    file = f'{save_dir}/p1-{args.problem1}-p2-{args.problem2}-i1-{args.instance1}-i2-{args.instance2}-run-{args.run}.csv'
    
    if os.path.isfile(file) == False:
        problem1 = get_problem(f"bbob-f{args.problem1}-{args.instance1}", n_var=args.dim)
        problem2 = get_problem(f"bbob-f{args.problem2}-{args.instance2}", n_var=args.dim)

        dfs = []
        for alpha in tqdm(np.linspace(0, 1, num=args.alphas)):
            mix = CombinedProblem(problem1, problem2, alpha)

            sampling = LHS()
            X = sampling(mix, args.dim*args.n_eval).get("X")
            y = mix.evaluate(X).flatten()
            
            features = {}
            
            features['problem1'] = args.problem1
            features['problem2'] = args.problem2
            features['instance1'] = args.instance1
            features['instance2'] = args.instance2
            features['alpha'] = alpha
            features['dim'] = args.dim
            features['algorithm_run'] = args.run
            
            methods = [
                calculate_cm_angle, 
                calculate_cm_conv, 
                calculate_cm_grad, 
                calculate_dispersion, 
                calculate_ela_conv, 
                calculate_ela_curvate, 
                calculate_ela_distribution, 
                calculate_ela_level, 
                calculate_ela_local, 
                calculate_ela_meta, 
                calculate_information_content, 
                calculate_limo, 
                calculate_nbc, 
                calculate_pca, 
                
            ]
            
            for method in methods:
                try:
                    fe = method(X, y)
                    features.update(fe)
                except:
                    pass
            
            #ela_meta = calculate_ela_meta(X, y)
            #features.update(ela_meta)
            #ela_distr = calculate_ela_distribution(X, y)
            #features.update(ela_distr)
            #ela_level = calculate_ela_level(X, y)
            #features.update(ela_level)
            #nbc = calculate_nbc(X, y)
            #features.update(nbc)
            #disp = calculate_dispersion(X, y)
            #features.update(disp)
            #ic = calculate_information_content(X, y)
            #features.update(ic)
            
            dfs.append(features)

        dfs = pd.DataFrame(dfs)
        dfs.to_csv(file, index=False)

