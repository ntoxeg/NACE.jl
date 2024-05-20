import jax
import xminigrid


key = jax.random.PRNGKey(0)
key, reset_key = jax.random.split(key)

env, env_params = xminigrid.make("MiniGrid-Empty-8x8")
print("Observation shape:", env.observation_shape(env_params))
print("Num actions:", env.num_actions(env_params))
