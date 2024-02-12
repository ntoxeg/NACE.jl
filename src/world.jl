export Cell, parse_world, world1, world2

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

"""Parse characters into Cells"""
function parse_char(c::Char) :: Cell
    char_to_cell = Dict(
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
    return char_to_cell[c]
end

"""Holds the ground-truth representation of the map"""
Map = Matrix{Cell}

"""
    parse_world(world_str::String) :: Map

Parse a string representation of the world into a `Map` of `Cell` values.
"""
function parse_world(world_str::String) :: Map
    # Split the world string into lines, skip the last character (newline)
    lines = split(world_str[begin:end-1], "\n")
    
    # Initialize an empty matrix to hold the parsed world
    world_matrix = Map(undef, length(lines), length(lines[1]))
    
    # Iterate over each line and character to parse the world
    for (i, line) in enumerate(lines)
        for (j, char) in enumerate(line)
            # Convert the character to the corresponding Cell enum value
            world_matrix[i, j] = parse_char(char)
        end
    end
    
    return world_matrix
end

# Example worlds
# TODO: process these with a macro.
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
