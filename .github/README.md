# SynerVQA GitHub Configuration

This folder contains all GitHub-specific configuration for the SynerVQA project.

## 📁 Structure

```
.github/
├── workflows/              # GitHub Actions CI/CD pipelines
│   ├── ci.yml             # Continuous Integration (tests, lint)
│   ├── train.yml          # Model training workflow
│   ├── evaluate.yml       # Model evaluation workflow
│   └── lint.yml           # Code quality checks
├── ISSUE_TEMPLATE/        # Issue templates for bug reports, features, etc.
├── PULL_REQUEST_TEMPLATE/ # PR template for contributions
├── CODEOWNERS             # Auto-assign reviewers
├── dependabot.yml         # Automated dependency updates
├── copilot-instructions.md # GitHub Copilot context
└── FUNDING.yml            # Sponsorship configuration
```

## 🚀 Workflows

### CI Pipeline (`ci.yml`)
Runs on every push and PR:
- Linting with flake8
- Formatting check with black
- Type checking with mypy
- Unit tests with pytest
- Coverage reporting

### Training Workflow (`train.yml`)
Manual trigger to train models:
```bash
gh workflow run train.yml -f model=stream_a -f epochs=10
```

### Evaluation Workflow (`evaluate.yml`)
Manual trigger to evaluate models:
```bash
gh workflow run evaluate.yml -f checkpoint=checkpoints/best.pth
```

## 🐛 Issue Templates

Available templates:
- **Bug Report**: For reporting bugs
- **Feature Request**: For suggesting new features
- **Model Performance**: For model accuracy issues
- **Dataset Issue**: For data pipeline problems

## 📝 Contributing

When creating a PR, the template will guide you through:
- Description of changes
- Type of change
- Testing performed
- Performance impact
- Checklist of requirements

## 🤖 Copilot

The `copilot-instructions.md` file provides context to GitHub Copilot about:
- Project structure
- Coding conventions
- Common patterns
- PyTorch best practices

## 🔄 Dependabot

Automatically creates PRs for:
- Python package updates (weekly)
- GitHub Actions updates (weekly)
- Security patches (immediate)

## 👥 Code Owners

The `CODEOWNERS` file ensures proper review for:
- Model architecture changes
- Data pipeline modifications
- Configuration updates
- Documentation changes

---

**Last Updated**: December 2024
