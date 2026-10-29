# Data Card - Hyperparameter Optimizer

The project bundles standard synthetic objective functions, not a dataset. There is no
training data and nothing is downloaded.

## benchmark objectives (`src/hyperparameter_optimizer/objectives.py`)

- **Provenance:** the classic black-box optimization test functions - sphere, Branin,
  Rastrigin, and Ackley - each with its standard domain and published global minimum.
- **Why these:** they span easy (sphere), moderately deceptive (Branin, Ackley), and
  pathologically multimodal (Rastrigin), so a sampler's behaviour is measured honestly.
- **Licensing:** standard mathematical functions, implemented in this repository; MIT.
- **Known gaps:** all are low-dimensional; they do not represent high-dimensional tuning.

## Ethical considerations

The objectives are mathematical functions with no personal, sensitive, or copyrighted content.
