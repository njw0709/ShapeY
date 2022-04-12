import subprocess
import os

python_version = "python3.6"

#runs on base shapey
networks = [
    'tf_efficientnet_l2_ns_475',
    'resnetv2_101x3_bitm',
    'resnetv2_101x3_bitm_in21k',
    'SimCLR_ResNet50_4x', #needs to be re-run
]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    subprocess.run([python_version, 
                    'analysis.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR)])

    subprocess.run([python_version,
                    'graph.py',
                    '-m',
                    'network={}'.format(','.join(networks)),
                    'data.project_dir={}'.format(PROJECT_DIR)])

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
                    'data.project_dir={}'.format(PROJECT_DIR)])