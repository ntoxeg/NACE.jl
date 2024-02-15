export Env, EnvParams, BuiltinEnv, register_env, make, render

abstract type Env end

struct EnvParams
    observation_shape::Tuple{Int,Int}
    action_size::Int
end

struct BuiltinEnv <: Env
    begin_state::Map
    state::Map
    params::EnvParams
end

REGISTERED_ENVIRONMENTS::Dict{String,Type} = Dict()

"""
    register_env(name::String, envT::Type{E}) where E <: Env

Add an environment to the global registry
"""
function register_env(name::String, envT::Type{E}) where E<:Env
    REGISTERED_ENVIRONMENTS[name] = envT
end

function make_registered(::Type{BuiltinEnv}, name::String)::Env
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

function make_registered(::Type{Env}, name::String)::Env
    error("Not implemented yet.")
end

"""
    make(name::String) :: Tuple{Env,EnvParams}

Create an instance of a registered environment
"""
function make(name::String)::Env
    if haskey(REGISTERED_ENVIRONMENTS, name)
        envT = REGISTERED_ENVIRONMENTS[name]
        return make_registered(envT, name)
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
