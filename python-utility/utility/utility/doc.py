from . import grouper

def cube():
    return """
         +--------+
        / axis 0 /|
       /  top   / |
      /        /  |
     /        /   |
    +--------+ax 2|
    |        |side|
    | axis 1 |    +
    | front  |   /
    |        |  /
    |        | /
    |        |/
    +--------+
    """

def results(*args, compact=False, start=None, end=None):
    """Print results in documentation"""
    if start is not None:
        print(start)
        if not compact:
            print()
    groups = grouper(args, 2)
    for i, (s, o) in enumerate(grouper(args, 2)):
        if compact:
            print(s, end=" ")
        else:
            print(s)
        print(o)
        if i < len(groups) - 1 and not compact:
            print()
    if end is not None:
        print(end)