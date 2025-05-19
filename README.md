Install the following tools on your Mac:

# Package manager
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Core tools
brew install docker
brew install kubectl
brew install minikube
brew install helm
brew install kind
brew install jq
brew install mkcert # for HTTPS
brew install node

# Optional for GUI-based logs and YAMLs
brew install lens
brew install k9s

# Python tools
brew install pyenv
brew install poetry

