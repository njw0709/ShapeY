import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# You must implement 'step1_save_feature/your_feature_extraction_code.py' before you run this script.
python_version = "python3.6"

networks = [
    'tf_efficientnet_l2_ns_475',
    'resnetv2_50x1_bitm',
    'cait_s24_224',
    'convit_base',
    'convnext_base',
    'deit_base_distilled_patch16_224',
    'levit_384'
]

for feature_extraction_system_name in networks:
    
    # saves the embedding vectors of the images to .h5 file.
    subprocess.run([python_version, 
                    os.path.join("step1_save_feature", "save_feature2h5py.py"), 
                    "--input_dir", 
                    os.path.join("data", "ShapeY200"),
                    "--output_dir",
                    os.path.join("data", "intermediate"),
                    "--run_example", 
                    "0",
                    "--name",
                    feature_extraction_system_name,
                    ])

    # computes correlation with the extracted embedding vectors.
    subprocess.run([python_version,
                    os.path.join("step2_compute_feature_correlation", "compute_correlation.py"),
                    "--input_dir",
                    os.path.join("data", "intermediate", feature_extraction_system_name+".h5"),
                    "--batch_size", "20000",
                    "--feature_name", feature_extraction_system_name,
                    "--num_workers", "8"
                    ])

    # runs nearest-neighbor benchmark analaysis
    subprocess.run([python_version,
                    os.path.join("step3_benchmark_analysis", "get_nn_classification_error_with_exclusion_distance.py"),
                    "--input_dir",
                    os.path.join("data", "intermediate", feature_extraction_system_name+".h5"),
                    "--output_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+".h5")
                    ])

    # graphs results (exclusion distance vs nn matching error)
    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+".h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "0"
                    ])

    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+".h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "1"
                    ])

