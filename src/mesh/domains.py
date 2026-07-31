import math


def genpolygon():
    R = 1.0
    r = R / (math.sin(math.pi / 5.) * math.tan(2 * math.pi / 5.) + math.cos(math.pi / 5.))
    vertices = []
    segments = []
    for n in range(10):
        angle = math.pi / 2 + n * math.pi / 5
        if n % 2 == 0:
            radius = R
        else:
            radius = r
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        vertices.append([x, y])
        segments.append([n, n + 1])
    vertices.append(vertices[0])
    segments.append([9, 0])
    return vertices, segments
