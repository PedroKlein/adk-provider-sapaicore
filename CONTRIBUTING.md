# Contributing

Contributions are welcome! Here's how to get started.

## Development

```bash
go build ./...
go vet ./...
golangci-lint run ./...
go test -race ./...
```

## Submitting changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Make your changes with tests
4. Ensure all checks pass locally
5. Submit a pull request

## Guidelines

- Follow existing code style (enforced by golangci-lint)
- Add tests for new functionality
- Keep PRs focused — one feature or fix per PR
- Update the README if you change the public API

## Reporting bugs

Open an issue with:
- Go version and OS
- Minimal reproduction steps
- Expected vs actual behavior
