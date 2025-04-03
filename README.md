# nnutils-2x

Neural network utilities compatible with PyTorch 2.x.

## Features

- `adaptive_avgpool_2d`: 2D adaptive average pooling with support for variable-sized inputs
- `adaptive_maxpool_2d`: 2D adaptive max pooling with support for variable-sized inputs
- `mask_image_from_size`: Mask a batch of images based on individual sizes

## Installation

```bash
pip install nnutils-2x
```

Or install from source:

```bash
git clone https://github.com/username/nnutils-2x.git
cd nnutils-2x
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 2.0+

## Example Usage

```python
import torch
from nnutils import adaptive_avgpool_2d, adaptive_maxpool_2d, mask_image_from_size

# Create a batch of 3 images with different sizes
batch = torch.randn(3, 3, 64, 64)
batch_sizes = torch.tensor([[32, 48], [64, 64], [48, 32]])

# Apply adaptive average pooling
avg_pooled = adaptive_avgpool_2d(batch, (16, 16), batch_sizes)

# Apply adaptive max pooling
max_pooled = adaptive_maxpool_2d(batch, (16, 16), batch_sizes)

# Mask images according to their sizes
masked = mask_image_from_size(batch, batch_sizes, mask_value=0)
```

## Development

### Directory Structure

```
nnutils-2x/
├── nnutils/           # Main package source code
├── tests/             # Test files
├── setup.py           # Package setup script
├── pytest.ini         # Pytest configuration
└── run_tests.py       # Script to run tests
```

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
python run_tests.py

# Or using pytest directly
pytest
```

## License

MIT
