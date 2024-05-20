export AbstractEnv, EnvParams, BuiltinEnv, register_env, make, render
include("world.jl")

abstract type AbstractEnv end

struct EnvParams
    observation_shape::Tuple{Int,Int}
    action_size::Int
end

struct BuiltinEnv <: AbstractEnv
    begin_state::Dict
    state::Dict
    params::EnvParams
end

REGISTERED_ENVIRONMENTS::Dict{String,Type} = Dict()

"""
    register_env(name::String, envT::Type{E}) where E <: Env

Add an environment to the global registry
"""
function register_env(name::String, envT::Type{E}) where E<:AbstractEnv
    REGISTERED_ENVIRONMENTS[name] = envT
end

function _make(::Type{BuiltinEnv}, name::String)::AbstractEnv
    if name == "Builtin-1"
        envp = EnvParams((3, 3), 4)
        env = BuiltinEnv(world1, world1, envp)
        return env

    elseif name == "Builtin-2"
        envp = EnvParams((3, 3), 4)
        env = BuiltinEnv(world2, world2, envp)
        return env

    elseif name == "Empty-10x10"
        envp = EnvParams((3, 3), 4)
        env = BuiltinEnv(empty_10x10, empty_10x10, envp)
        return env
    else
        error("Unknown environment '$name'!")
    end
end

function _make(::Type{AbstractEnv}, name::String)::AbstractEnv
    error("Not implemented yet.")
end

"""
    make(name::String) :: Tuple{Env,EnvParams}

Create an instance of a registered environment
"""
function make(name::String)::AbstractEnv
    if haskey(REGISTERED_ENVIRONMENTS, name)
        envT = REGISTERED_ENVIRONMENTS[name]
        return _make(envT, name)
    else
        error("Unknown environment '$name'!")
    end
end

"""
    render(env::BuiltinEnv)

Render a built-in environment
"""
function render(env::BuiltinEnv)
    print(show_world(env.state))
end

### Registered environments ###
register_env("Empty-10x10", BuiltinEnv)
register_env("Builtin-1", BuiltinEnv)
register_env("Builtin-2", BuiltinEnv)

### minigrid stuff
# Map of object type to integers
OBJECT_TO_IDX = Dict(
    "unseen" => 0,
    "empty" => 1,
    "wall" => 2,
    "floor" => 3,
    "door" => 4,
    "key" => 5,
    "ball" => 6,
    "box" => 7,
    "goal" => 8,
    "lava" => 9,
    "agent" => 10
)

IDX_TO_OBJECT = Dict(value => key for (key, value) in OBJECT_TO_IDX)

# Map of state names to integers
STATE_TO_IDX = Dict(
    "open" => 0,
    "closed" => 1,
    "locked" => 2
)

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    [1, 0],
    # Down (positive Y)
    [0, 1],
    # Pointing left (negative X)
    [-1, 0],
    # Up (negative Y)
    [0, -1]
]


