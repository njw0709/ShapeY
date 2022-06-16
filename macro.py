import subprocess
import os

python_version = "python3.6"

#runs on base shapey
networks = [
    'example'
]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    subprocess.run([python_version,
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR), 
                    'data=ShapeY200'])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR),
                    'graph=obj,cat'])

    #Contrast Reversed
    subprocess.run([python_version,
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200CR',
                    'data.project_dir={}'.format(PROJECT_DIR)])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data=ShapeY200CR',
                    'data.project_dir={}'.format(PROJECT_DIR),
                    'graph=obj_cr_hard,obj_cr_soft,cat_cr_hard,cat_cr_soft'])