lint: ## run the lint checkers
	uv run ruff check gym_AO/envs/AO_env_artiom.py    
	uv run ruff format gym_AO/envs/AO_env_artiom.py