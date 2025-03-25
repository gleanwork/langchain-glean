# Release Process

This document outlines the steps to create and publish a new release of `langchain-glean`.

## Prerequisites

- You have push access to the main repository
- You have PyPI publishing rights (via Trusted Publisher workflow)
- You have [Task](https://taskfile.dev) installed (`brew install go-task`)

## Preview Changes

Before making a release, you can preview what would happen:

```bash
task release DRY_RUN=true
```

This will show you:
- What the next version would be
- What changes would be included in the changelog
- What files would be modified

No files are actually changed during preview.

## Release Steps

1. **Ensure your local main is up to date**
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Run tests and checks**
   ```bash
   task test:all
   ```

3. **Bump version and generate changelog**
   ```bash
   task release
   ```
   This command will:
   - Bump the version based on your commit history
   - Generate/update CHANGELOG.md in the root directory
   - Create a new commit with these changes
   - Create a new git tag (e.g., v0.1.1)

4. **Push changes and tag**
   ```bash
   git push origin main --tags
   ```

The GitHub Action will automatically:
- Build the package
- Create a GitHub release with the changelog content
- Publish to PyPI using trusted publishing

## Verifying the Release

1. Check the GitHub Actions tab to ensure the publish workflow completed successfully
2. Verify that:
   - CHANGELOG.md has been updated in the repository
   - A new GitHub release was created with the changelog content
   - The new version tag is visible in GitHub
3. Verify the new version is available on [PyPI](https://pypi.org/project/langchain-glean/)

## Troubleshooting

If the release fails:

1. Check the GitHub Actions logs for any errors
2. If needed, delete the tag locally and remotely:
   ```bash
   git tag -d v0.1.1
   git push --delete origin v0.1.1
   ```
3. Delete the GitHub release if it was created
4. Fix any issues and retry from step 1

## Notes

- The version bump follows [Semantic Versioning](https://semver.org/)
- The changelog is automatically generated in CHANGELOG.md from your commit messages
- We use [Commitizen](https://commitizen-tools.github.io/commitizen/) for version management
- GitHub releases are automatically created with changelog content
- The release process is automated via GitHub Actions using PyPI's trusted publisher workflow 