export Env, EnvParams, BuiltinEnv, register_env, make

abstract type Env end

struct EnvParams
    observation_shape::Tuple{Int,Int}
    action_size::Int
end

struct BuiltinEnv <: Env
    begin_state::Map
    state::Map
end

# TODO: figure out the type.
REGISTERED_ENVIRONMENTS::Dict = Dict()

function register_env(name::String, envT::Type{E}) where E <: Env
    REGISTERED_ENVIRONMENTS[name] = envT
end


function make_builtin(name::String, envT::Type{BuiltinEnv}) :: Tuple{Env,EnvParams}
    if name == "Builtin-1"
        env = BuiltinEnv(world1, world1)
        envp = EnvParams((3, 3), 4)
        return env, envp
        
    elseif name == "Builtin-2"
        env = BuiltinEnv(world2, world2)
        envp = EnvParams((3, 3), 4)
        return env, envp
    else
        error("Unknown environment '$name'.")
    end
end

function make_extern(name::String, envT::Type{BuiltinEnv}) :: Tuple{Env,EnvParams}
    error("Not implemented yet.")
end

function make(name::String) :: Tuple{Env,EnvParams}
    if haskey(REGISTERED_ENVIRONMENTS, name)
        envT = REGISTERED_ENVIRONMENTS[name]
        if isa(envT, Type{BuiltinEnv})
            return make_builtin(name, envT)
        else
            return make_extern(name, envT)
        end
    else
        error("Unknown environment '$name'.")
    end
end

register_env("Builtin-1", BuiltinEnv)
register_env("Builtin-2", BuiltinEnv)
