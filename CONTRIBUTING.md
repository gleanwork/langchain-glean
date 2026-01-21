# Contributing to langchain-glean

This document provides guidelines and instructions for setting up your development environment and contributing to `langchain-glean`.

## Development Environment

This project uses [mise](https://mise.jdx.dev/) for tool management and task running, and [uv](https://github.com/astral-sh/uv) for dependency management.

### Prerequisites

1. Install mise:

    ```bash
    curl https://mise.run | sh
    ```

    Or via package manager (see [mise installation docs](https://mise.jdx.dev/getting-started.html)).

### Setup Development Environment

1. Install tools and set up the development environment:

    ```bash
    mise install
    mise run setup
    ```

This will install Python and uv via mise, then create a virtual environment and install all dependencies.

## Development Tasks

The project uses [mise tasks](https://mise.jdx.dev/tasks/) to manage development tasks. Here are the available tasks:

### Testing

| Task | Description |
|------|-------------|
| `mise run test` | Run unit tests |
| `mise run test:watch` | Run tests in watch mode |
| `mise run test:integration` | Run integration tests |
| `mise run test:all` | Run all tests and lint fixes |

### Linting and Formatting

| Task | Description |
|------|-------------|
| `mise run lint` | Run all linters |
| `mise run lint:diff` | Run linters on changed files |
| `mise run lint:package` | Run linters on package files |
| `mise run lint:tests` | Run linters on test files |
| `mise run lint:fix` | Run lint autofixers |
| `mise run lint:fix:diff` | Run lint autofixers on changed files |
| `mise run lint:fix:package` | Run lint autofixers on package files |
| `mise run lint:fix:tests` | Run lint autofixers on test files |
| `mise run format` | Run code formatters |
| `mise run format:diff` | Run formatters on changed files |

### Utility Tasks

| Task | Description |
|------|-------------|
| `mise run spell:check` | Check spelling |
| `mise run spell:fix` | Fix spelling |
| `mise run check:imports` | Check imports |

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Make your changes and ensure that all tests pass.
3. Update the documentation to reflect any changes.
4. Submit a pull request.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.
