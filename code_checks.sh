black --check src
black --check scripts
flake8 src
#flake8 scripts
mypy --ignore-missing-imports src
