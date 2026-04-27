# globals_state.py

cancel_requested = False
cancel_delete_downloads = False  # if True, delete completed model downloads on cancel
ram_tier_drop = False      # set True when RAM drops enough to compromise the current tier

current_tps = 0
current_task = "Initializing..."
current_prompt_index = 0
current_prompt_type = ""
current_token_count = None
total_prompts = 0

current_tier = None        # name of the tier currently being benchmarked

filename = ""
