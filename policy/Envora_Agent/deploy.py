"""demo_policy's deployment loop plus one ``begin_batch`` call after the reset: it
tells the agent the episode's world count, task config and arm mounts, which the
reset does not carry. The debug client has none of them; the server then falls
back to deploy.yml (task_name, robot_origin_poses)."""


def _batch_info(TASK_ENV, num_envs):
    robot_manager = getattr(TASK_ENV, "robot_manager", None)
    poses = None
    if robot_manager is not None:
        poses = [
            [float(value) for value in robot.entity_origin_pose]
            for robot in robot_manager.robot_list
            if getattr(robot, "type", "target") == "target"
        ]
    return {
        "num_envs": getattr(TASK_ENV, "num_envs", num_envs),
        "task_name": getattr(TASK_ENV, "task_name", None),
        "robot_origin_poses": poses,
    }


def eval_one_episode(TASK_ENV, model_client):

    model_client.call(func_name="reset") # reset policy
    model_client.call(func_name="begin_batch", obs=_batch_info(TASK_ENV, 1))

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)

            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break

            obs = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    model_client.call(func_name="reset")
    model_client.call(
        func_name="begin_batch",
        obs=_batch_info(TASK_ENV, len(TASK_ENV.get_running_env_idx_list())),
    )

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = TASK_ENV.get_running_env_idx_list() # Get Running Environment Index List
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Observation
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)  # Get Action according to observation chunk

        chunk_size = len(actions[0]) # Get the chunk size
        for action_idx in range(chunk_size): # Iterate over the action chunk
            current_action_list = [env_actions[action_idx] for env_actions in actions] # Get the current action list
            TASK_ENV.take_action_batch(current_action_list, env_idx_list) # Take the action

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size: # Check whether the episode ends
                break

            running = set(TASK_ENV.get_running_env_idx_list()) # Get the running environment index list
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running] # Get the active batch index

            actions = [actions[i] for i in active_batch_idx] # Get the active action list
            env_idx_list = [env_idx_list[i] for i in active_batch_idx] # Get the active environment index list
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list)) # Update the observation
