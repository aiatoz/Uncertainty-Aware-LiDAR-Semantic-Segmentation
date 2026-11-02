# Uncertainty-Aware Semantic Segmentation Pipelines for LiDAR Point Clouds

This is the codebase for the paper "Role-based Effectiveness of an Epistemic Uncertainty Estimate in Training Models for Semantic Segmentation of 3D LiDAR Point Clouds". It is advised to run the models either on Linux or WSL, otherwise library compatibility issues may arise.

# Notes:
- Since our custom PNPP pipeline reuses some of the modules and intermediate files generated from the RandLA-Net pipeline, please follow the execution order: ANN, RandLA-Net, PNPP.
- We have uploaded the code and the respective guidelines; please follow them to reproduce the results. Due to space limitations, we have not included the datasets and resulting files in this repository. The datasets can be downloaded from their official repositories or websites (these files were converted to CSV format using CloudCompare: https://www.cloudcompare.org/):

    - AHN3: https://geotiles.citg.tudelft.nl/
    - Vaihingen: https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx or https://www.kaggle.com/datasets/debobratachakraborty/vaihingen-point-cloud
    - NPM3D / Paris-Lille-3D: https://npm3d.fr/paris-lille-3d
    - Toronto-3D: https://github.com/WeikaiTan/Toronto-3D
