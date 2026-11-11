"""Round-trip check of the Python binding (run by cibuildwheel against each built wheel)."""

import array

import volt_spatial_assembler as sa

positions = array.array('f', [0, 0, 0, 1, 2, 3, 640, 630, 340])
[points] = sa.decode_glb(sa.encode_point_cloud(positions, sa.colors_by_type(array.array('H', [1, 2, 9]))))
decoded = array.array('f')
decoded.frombytes(points['positions'])
assert points['mode'] == 'points'
assert all(abs(a - b) <= sa.POSITION_STEP / 2 for a, b in zip(decoded, positions))

[mesh] = sa.decode_glb(sa.encode_mesh(
    array.array('f', [0, 0, 0, 1, 0, 0, 0, 1, 0]),
    array.array('f', [0, 0, 1] * 3),
    array.array('I', [0, 1, 2]),
))
assert mesh['mode'] == 'triangles' and len(mesh['indices']) == 3 * 4
print('volt_spatial_assembler round-trip ok')
