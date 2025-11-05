SHELL := /bin/bash
ENV_NAME := ligo
ENV_FILE := environment.yml

.PHONY: env html clean

env:
	conda env update -n $(ENV_NAME) -f $(ENV_FILE) || conda env create -n $(ENV_NAME) -f $(ENV_FILE)
	@echo "Environment '$(ENV_NAME)' ready."

html:
	myst build --html
	@echo "Built HTML site in _build/html/."

clean:
	rm -rf figures/* audio/* _build
	@echo "Cleaned generated files."