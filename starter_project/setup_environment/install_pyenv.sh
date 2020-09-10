# installing pyenv, which is my preferred toolkit for managing Python versions and packages

echo "Installing dependencies for pyenv on Debian distribution of Linux..."
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl

echo "Installing pyenv..."
curl https://pyenv.run | bash


echo "Configuring bashrc..."
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo "Reloading shell..."
exec "$SHELL"
