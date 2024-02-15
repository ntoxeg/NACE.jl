export Cell, parse_world, show_world, world1, world2, empty_10x10

@enum Cell begin
    WALL
    ROBOT
    CUP
    FOOD
    BATTERY
    FREE
    TABLE
    GOAL
    KEY
    DOOR
    ARROW_DOWN
    ARROW_UP
    BALL
    EGG
    EGGPLACE
    CHICKEN
    SBALL
    SHOCK
end

char_to_cell::Dict{Char,Cell} = Dict(
    'o' => WALL,
    'x' => ROBOT,
    'u' => CUP,
    'f' => FOOD,
    'b' => BATTERY,
    ' ' => FREE,
    'T' => TABLE,
    'H' => GOAL,
    'k' => KEY,
    'D' => DOOR,
    'v' => ARROW_DOWN,
    '^' => ARROW_UP,
    'c' => BALL,
    'O' => EGG,
    '_' => EGGPLACE,
    '4' => CHICKEN,
    '0' => SBALL,
    'z' => SHOCK,
)
"""
Parse a character into a Cell
"""
parse_char(c::Char)::Cell = char_to_cell[c]

cell_to_char::Dict{Cell,Char} = Dict(
    WALL => 'o',
    ROBOT => 'x',
    CUP => 'u',
    FOOD => 'f',
    BATTERY => 'b',
    FREE => ' ',
    TABLE => 'T',
    GOAL => 'H',
    KEY => 'k',
    DOOR => 'D',
    ARROW_DOWN => 'v',
    ARROW_UP => '^',
    BALL => 'c',
    EGG => 'O',
    EGGPLACE => '_',
    CHICKEN => '4',
    SBALL => '0',
    SHOCK => 'z',
)
"""
Turn a Cell into a character
"""
show_cell(cell::Cell)::Char = cell_to_char[cell]

"""
Holds the ground-truth representation of the map
"""
Map = Matrix{Cell}

"""
    parse_world(world_str::String) :: Map

Parse a string representation of the world into a `Map` of `Cell` values.
"""
function parse_world(world_str::String)::Map
    # Split the world string into lines, skip the last character (newline)
    lines = split(world_str[begin:end-1], "\n")

    # Initialize an empty matrix to hold the parsed world
    world_matrix = Map(undef, length(lines), length(lines[1]))

    # Iterate over each line and character to parse the world
    for (i, line) ∈ enumerate(lines)
        for (j, char) ∈ enumerate(line)
            # Convert the character to the corresponding Cell enum value
            world_matrix[i, j] = parse_char(char)
        end
    end

    world_matrix
end

"""
    show_world(world::Map) :: String

Turn a world matrix into a string
"""
function show_world(world::Map)::String
    world_str = ""
    for row ∈ eachrow(world)
        row_str = map(show_cell, row)
        push!(row_str, '\n')
        world_str *= join(row_str)
    end
    world_str
end

# Example worlds
# TODO: process these with a macro.
empty_10x10 = parse_world("""
oooooooooo
o        o
o        o
o        o
o        o
o    x   o
o        o
o        o
o        o
oooooooooo
""")

world1 = parse_world("""
oooooooooooo
o   o   f  o
o          o
o   oooooooo
o x        o
o       u  o
oooooooooooo
""")

world2 = parse_world("""
oooooooooooo
o          o
o   u      o
o     ooooTo
o x        o
o          o
oooooooooooo
""")
