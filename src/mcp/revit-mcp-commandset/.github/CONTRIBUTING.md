# Contributing Guide

Welcome to contribute! Please read this guide before participating.

## Development Environment Setup

1. Make sure you have the following software installed:
   - Visual Studio 2019 or higher
   - .NET Framework 4.8 (or the version specified by the project)
   - Revit API (depending on the target Revit version for the project)

2. Clone the repository:
   ```bash
   git clone https://github.com/revit-mcp/revit-mcp-commandset.git
   ```

3. Open the solution file `revit-mcp-commandset.sln` in Visual Studio

## Code Contribution Process

1. Create a new branch from the latest `main` branch
2. Develop your feature or fix
3. Ensure the code compiles successfully and passes all tests
4. Commit with clear messages: [Consider following Conventional Commit Guidlines](https://www.conventionalcommits.org/en/v1.0.0/#summary)
5. Push the branch and create a Pull Request

## Branch Naming Conventions

Branch naming should follow:
- Feature branches: `feature/your-feature-name`
- Fix branches: `fix/issue-description`
- Documentation branches: `docs/update-description`

## Code Style

- Follow C# coding conventions and existing code style
- Use meaningful variable and function names
- Add appropriate comments and documentation strings
- Run code analysis tools before committing

## Commit Messages

Use conventional commit formats:

```
feature: Add new feature
fix: Fix bug
docs: Update documentation
refactor: Refactor code (no feature changes)
test: Add tests
```

## Reporting Issues

- Search for existing issues before creating a new one
- Provide detailed reproduction steps and environment information, including:
  - Revit version
  - Visual Studio version
  - Operating system version
  - Error logs (if available)

## Code of Conduct

Please adhere to our [Code of Conduct](CODE_OF_CONDUCT.md)
