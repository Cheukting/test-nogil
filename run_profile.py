import cProfile, pstats

import sys, os

import importlib

file_path = sys.argv[1]

import_path = '/'.join(file_path.split('/')[:-1])
module_name = file_path.split('/')[-1].split('.')[0]
print(f"Running {module_name} from {import_path} using {sys.version}")
sys.path.append(os.path.abspath(import_path))

imported = importlib.import_module(module_name)


profiler = cProfile.Profile()
profiler.enable()

imported.run_all()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()

print('-'*28)
