# Dataset Notices & Licensing Restrictions

This project utilizes three distinct agricultural image datasets for research, cross-domain validation, and benchmarking of plant leaf disease classification architectures.

---

## Dataset Summary & Licensing

| Dataset | Primary Source | Intended Domain | License / Distribution Terms |
| :--- | :--- | :--- | :--- |
| **PlantVillage** | Hughes & Salathé (2015), [Kaggle abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | Controlled laboratory conditions (single leaf on uniform background) | **verify before redistribution** |
| **PlantDoc** | Singh et al. (2019), [GitHub pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) | In-the-wild field conditions (whole plants, complex backgrounds) | **verify before redistribution** |
| **PlantWild v1** | Agricultural wild collection, [Kaggle](https://www.kaggle.com/) | Real-world field captures, natural illumination, varied leaf stages | **CC BY-NC-ND 4.0 (non-commercial, no derivatives)** |

---

## Redistribution & Weight Sharing Restrictions

> [!CAUTION]
> **Trained model weights and dataset binaries must NOT be redistributed publicly.**

1. **PlantWild v1 Restrictions (`CC BY-NC-ND 4.0`)**:
   - The PlantWild v1 dataset is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License ([CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)).
   - This license expressly forbids commercial use and the public distribution of derivative works.
   - As machine learning model checkpoints fine-tuned or trained on PlantWild constitute derivative representations of the underlying data distribution, **trained weights and model artifacts incorporating PlantWild must not be made publicly accessible or commercially distributed**.

2. **PlantVillage & PlantDoc (`verify before redistribution`)**:
   - Both PlantVillage and PlantDoc originate from academic repositories and web scraping efforts where redistribution rights vary by jurisdiction and downstream terms.
   - Users must independently verify terms of service before attempting any redistribution of raw images.

3. **Code vs. Data Separation**:
   - The software, training loops, evaluation suites, and configurations in this repository are licensed under the MIT License (see [LICENSE](file:///c:/Major%20Project/plant-disease-detection/LICENSE)).
   - This software license does **not** grant rights to redistribute the underlying data or trained weights.
