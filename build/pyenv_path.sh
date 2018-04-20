#!/usr/bin/env bash
if [ $SHELL == "/bin/zsh" ]; then
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
  echo 'if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi' >> ~/.zshrc
  echo 'export PATH="/Users/username/.pyenv:$PATH"' >> ~/.zshrc
  source ~/.zshrc
else
  echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
  echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bash_profile
  echo 'if which pyenv-virtualenv-init > /dev/null; then eval "$(pyenv virtualenv-init -)"; fi' >> ~/.bash_profile
  echo 'export PATH="/Users/username/.pyenv:$PATH"' >> ~/.bash_profile
  source ~/.bash_profile
fi
