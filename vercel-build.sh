#!/bin/bash

set -e

HUGO_VERSION="0.164.0"

echo "Installing Hugo Extended ${HUGO_VERSION}..."

curl -L -o hugo.tar.gz \
  "https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_extended_${HUGO_VERSION}_linux-amd64.tar.gz"

tar -xzf hugo.tar.gz

mkdir -p "$HOME/bin"
mv hugo "$HOME/bin/hugo"

export PATH="$HOME/bin:$PATH"

echo "Hugo version:"
hugo version

echo "Building Hugo site..."
hugo --minify

echo "Running Pagefind..."
pnpm run pagefind
