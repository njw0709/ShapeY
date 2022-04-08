import subprocess
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# You must implement 'step1_save_feature/your_feature_extraction_code.py' before you run this script.
python_version = "python3.6"

networks = [
    'vit_huge_patch14_224_in21k',
    'vit_large_patch16_224',
    'xcit_large_24_p8_224',
    'xcit_large_24_p8_224_dist',
    'resnetv2_101x3_bitm',
    'resnetv2_101x3_bitm_in21k',
    'beit_base_patch16_224',
    'beit_base_patch16_224_in22k',
    'coat_mini',
    'mixer_b16_224',
    'mobilenetv3_large_100',
    'resnetv2_50x1_bit_distilled',
    'resnext50d_32x4d',
    'seresnet50',
    'swsl_resnet50'
]

# running contrast reversed
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
                    "--contrast_reversed",
                    "1",
                    '--contrast_reversed_input_dir',
                    os.path.join("data", "ShapeY200CR")
                    ])

    # computes correlation with the extracted embedding vectors.
    subprocess.run([python_version,
                    os.path.join("step2_compute_feature_correlation", "compute_correlation.py"),
                    "--input_dir",
                    os.path.join("data", "intermediate", feature_extraction_system_name+".h5"),
                    "--batch_size", "20000",
                    "--feature_name", feature_extraction_system_name,
                    "--num_workers", 
                    "8",
                    "--contrast_reversed",
                    "1"
                    ])

    # runs nearest-neighbor benchmark analaysis
    subprocess.run([python_version,
                    os.path.join("step3_benchmark_analysis", "get_nn_classification_error_with_exclusion_distance.py"),
                    "--input_dir",
                    os.path.join("data", "intermediate", feature_extraction_system_name+".h5"),
                    "--output_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+"_cr.h5"),
                    "--contrast_reversed",
                    "1"
                    ])

    # graphs results (exclusion distance vs nn matching error)
    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+"_cr.h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "0",
                    "--contrast_reversed",
                    "1",
                    "--exclusion_mode",
                    "soft"
                    ])

    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+"_cr.h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "1",
                    "--contrast_reversed",
                    "1",
                    "--exclusion_mode",
                    "soft"
                    ])

    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+"_cr.h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "0",
                    "--contrast_reversed",
                    "1",
                    "--exclusion_mode",
                    "hard"
                    ])

    subprocess.run([python_version,
                    os.path.join("step4_graph_results", "graph_exclusion_top1error_v2.py"),
                    "--input_dir",
                    os.path.join("data", "processed", feature_extraction_system_name+"_cr.h5"),
                    "--output_dir",
                    os.path.join("figures", feature_extraction_system_name),
                    "--within_category_error",
                    "1",
                    "--contrast_reversed",
                    "1",
                    "--exclusion_mode",
                    "hard"
                    ])
