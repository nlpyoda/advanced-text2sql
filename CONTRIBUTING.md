# Contributing to Advanced Text2SQL

We welcome contributions to the Advanced Text2SQL project! This document provides guidelines for contributing.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature/fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## 📋 Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/advanced-text2sql.git
cd advanced-text2sql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install in development mode
pip install -e .[dev]
```

## 🧪 Testing

Before submitting a PR, ensure:

```bash
# Run tests
pytest tests/

# Check code formatting
black --check .

# Run linting
flake8 .

# Type checking
mypy .
```

## 📝 Code Style

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://pep8.org/) guidelines
- Add type hints where appropriate
- Write docstrings for public functions/classes

## 🐛 Bug Reports

When filing bug reports, please include:

- Python version
- Package version
- Operating system
- GPU/hardware information
- Minimal reproduction example
- Error messages/stack traces

## 💡 Feature Requests

For new features:

- Check existing issues first
- Describe the use case
- Provide implementation suggestions
- Consider backward compatibility

## 🔄 Pull Request Process

1. **Branch naming**: Use descriptive names like `feature/policy-solver-improvement`
2. **Commits**: Use clear, descriptive commit messages
3. **Documentation**: Update README/docs if needed
4. **Tests**: Add tests for new functionality
5. **Review**: Address feedback from reviewers

## 📊 Performance Benchmarks

When contributing performance improvements:

- Include before/after benchmarks
- Test on standard datasets (Spider, BIRD)
- Document hardware used for testing
- Include training time comparisons

## 🏗️ Architecture Guidelines

### Adding New Components

When adding new training components:

```python
class NewComponent(nn.Module):
    """Brief description of the component.
    
    Args:
        config: Configuration object
        
    Example:
        >>> component = NewComponent(config)
        >>> result = component.forward(inputs)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, inputs):
        # Implementation
        pass
```

### Configuration Changes

New configuration options should:

- Have sensible defaults
- Be documented with comments
- Include validation where appropriate
- Be backward compatible

## 📚 Documentation

- Update README.md for major features
- Add docstrings to all public methods
- Include usage examples
- Update performance benchmarks

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help newcomers get started
- Share knowledge and insights
- Credit others for their contributions

## 📞 Getting Help

- Open GitHub issues for bugs/features
- Start GitHub discussions for questions
- Check existing issues before posting
- Provide detailed context

## 🏆 Recognition

Contributors will be:

- Added to the contributors list
- Credited in release notes
- Mentioned in paper acknowledgments (if applicable)

Thank you for contributing to Advanced Text2SQL! 🎉