
# A simple framework for fast concept erasure on Text-to-Image diffusion model

## Description
This project provides a framework for rapid concept erasure in Text-to-Image (T2I) diffusion models. By using a dual-objective approach with a knowledge distillation framework, the model can erase target concepts while preserving others, completing this process within 15 seconds. This method offers a solution that is both faster and more efficient than traditional techniques.

![show](text-clip-erasing\image\show.jpg)

## Key Features
- **Dual-Objective Concept Erasure**: Erases specified concepts without affecting unrelated concepts.
- **High Efficiency**: Completes concept removal up to 50~600 times faster than existing methods.
- **Text Encoder Focus**: By focusing on the text encoder, the method avoids complex model modifications.
- **Broad Applicability**: Can be used for artistic styles, specific objects, NSFW content, and IP concepts.

## Files
- `infer-csv.py`: Loads the trained model to generate images based on the prompt
- `load_save.py`: Utilities for loading and saving models and configurations.
- `train-clip.py`: Main script for running concept erasure based on `concept.csv`.

## Installation
Install the required packages with:
```bash
pip install -r requirements.txt
```

## Usage
1. The calibration set concept is in concept.csv`.

2. Run the concept erasure process:
   ```bash
   python train-clip.py --concept 'Van Gogh style' --lambda_reg 50 --epochs 20
   ```

3. (Optional) Load the trained model production picture :
   ```bash
   python infer-csv.py --dataset_path /home/data/art_prompts.csv --text_encoder_path networks/VanGoghstyle-epoch-19
   ```

## License
Refer to `LICENSE` for terms and conditions.
