# Contributing

Contributions are welcome! Here's how to get started.

## Getting Started

1. Fork the repository
2. Clone your fork and create a feature branch
3. Install [Task](https://taskfile.dev) for build automation (optional but recommended)

## Development

```bash
# Run tests
go test ./...

# Run tests with race detector
go test -race -short ./...

# Lint (requires golangci-lint v2)
golangci-lint run ./...

# Full check (format, vet, lint, test)
task check
```

## Pull Requests

1. Add tests for new functionality
2. Ensure `go test ./...` passes
3. Ensure `golangci-lint run ./...` is clean
4. Run `go mod tidy` if dependencies changed
5. Keep commits focused and use descriptive messages

## Code Style

- Follow standard Go conventions (`gofmt`)
- The project uses an extensive golangci-lint configuration — run the linter before submitting
- Table-driven tests with `testify/assert` are preferred

## Reporting Bugs

Open a GitHub issue with:
- Go version and OS
- Steps to reproduce
- Expected vs actual behavior
- Sample audio parameters (sample rates, channels, quality preset) if applicable

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
