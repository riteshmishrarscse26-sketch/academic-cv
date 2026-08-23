#!/bin/bash

set -e

HUGO_VERSION="0.164.0"
GO_VERSION="1.25.0"

echo "======================================"
echo "Installing Go ${GO_VERSION}..."
echo "======================================"

curl -L -o go.tar.gz \
  "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"

mkdir -p "$HOME/go"

tar -xzf go.tar.gz -C "$HOME/go" --strip-components=1

export GOROOT="$HOME/go"
export GOPATH="$HOME/go-work"
export PATH="$GOROOT/bin:$GOPATH/bin:$PATH"

echo "Go version:"
go version

echo "======================================"
echo "Installing Hugo Extended ${HUGO_VERSION}..."
echo "======================================"

curl -L -o hugo.tar.gz \
  "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"

tar -xzf hugo.tar.gz

mkdir -p "$HOME/bin"
mv hugo "$HOME/bin/hugo"

export PATH="$HOME/bin:$PATH"

echo "Hugo version:"
hugo version

echo "======================================"
echo "Building Hugo site..."
echo "======================================"

hugo --minify

echo "======================================"
echo "Running Pagefind..."
echo "======================================"

pnpm run pagefind

echo "======================================"
echo "BUILD SUCCESSFUL"
echo "======================================"
