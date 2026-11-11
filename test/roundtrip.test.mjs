// Round-trip tests of the encoder/decoder: node --test test/
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const sa = createRequire(import.meta.url)('..');

const sphere = (radius, center, rings, segments) => {
    const positions = [];
    const normals = [];
    const colors = [];
    const indices = [];
    for (let r = 0; r <= rings; r++) {
        const theta = (r / rings) * Math.PI;
        for (let s = 0; s <= segments; s++) {
            const phi = (s / segments) * 2 * Math.PI;
            const n = [Math.sin(theta) * Math.cos(phi), Math.sin(theta) * Math.sin(phi), Math.cos(theta)];
            positions.push(...n.map((v, a) => center[a] + radius * v));
            normals.push(...n);
            colors.push(r / rings, s / segments, 0.5, 1);
        }
    }
    for (let r = 0; r < rings; r++) {
        for (let s = 0; s < segments; s++) {
            const a = r * (segments + 1) + s;
            const b = a + segments + 1;
            indices.push(a, b, a + 1, a + 1, b, b + 1);
        }
    }
    return {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
        colors: new Float32Array(colors),
        indices: new Uint32Array(indices)
    };
};

test('points keep their order and land within half a grid step', () => {
    const count = 20000;
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < positions.length; i++) positions[i] = ((i * 7919) % 6007) * 0.1;
    const types = new Uint16Array(count).map((_, i) => i % 9);
    const colors = sa.colorsByType(types);

    const [points] = sa.decodeGLB(sa.encodePointCloud(positions, colors));
    assert.equal(points.mode, 'points');
    assert.equal(points.positions.length, positions.length);
    for (let i = 0; i < positions.length; i++) {
        assert.ok(Math.abs(points.positions[i] - positions[i]) <= sa.POSITION_STEP / 2 + 1e-4, `coordinate ${i}`);
    }
    assert.deepEqual(points.colors, colors);
    assert.deepEqual(Array.from(colors.subarray(8 * 4, 9 * 4)), Array.from(colors.subarray(7 * 4, 8 * 4)), 'types above 7 take the last colour');
});

test('float colours are stored as RGBA8', () => {
    const [points] = sa.decodeGLB(sa.encodePointCloud(new Float32Array([0, 0, 0, 1, 1, 1]), new Float32Array([1, 0, 0, 0, 0.5, 1])));
    assert.deepEqual(Array.from(points.colors), [255, 0, 0, 255, 0, 128, 255, 255]);
});

test('meshes keep their geometry, normals and material', () => {
    const { positions, normals, colors, indices } = sphere(40, [300, 200, 100], 60, 120);
    const material = { baseColor: [0.8, 0.2, 0.1, 1], metallic: 0.1, roughness: 0.6, emissive: [0, 0, 0], doubleSided: false };
    const glb = sa.encodeMesh(positions, normals, indices, colors, material);
    assert.ok(glb.length < (positions.byteLength + normals.byteLength + colors.byteLength + indices.byteLength) / 5);

    const [mesh] = sa.decodeGLB(glb);
    assert.equal(mesh.mode, 'triangles');
    assert.equal(mesh.indices.length, indices.length);
    for (let i = 0; i < mesh.positions.length / 3; i++) {
        const d = [0, 1, 2].map((a) => mesh.positions[i * 3 + a] - [300, 200, 100][a]);
        const r = Math.hypot(...d);
        assert.ok(Math.abs(r - 40) <= sa.POSITION_STEP, `vertex ${i} radius ${r}`);
        const cos = d.reduce((sum, v, a) => sum + v * mesh.normals[i * 3 + a], 0) / r;
        assert.ok(cos > Math.cos((2 * Math.PI) / 180), `vertex ${i} normal`);
    }
    assert.equal(mesh.material.doubleSided, false);
    assert.ok(Math.abs(mesh.material.roughness - 0.6) < 1e-6);
});

test('empty inputs give a valid model with nothing in it', () => {
    assert.deepEqual(sa.decodeGLB(sa.encodePointCloud(new Float32Array(0))), []);
    assert.deepEqual(sa.decodeGLB(sa.encodeMesh(new Float32Array(0), new Float32Array(0), new Uint32Array(0))), []);
});

test('large extents fall back to a coarser uniform grid', () => {
    const [points] = sa.decodeGLB(sa.encodePointCloud(new Float32Array([0, 0, 0, 2000, 10, 10])));
    assert.ok(Math.abs(points.positions[3] - 2000) <= 2000 / 65535);
});

test('bad inputs are rejected', () => {
    assert.throws(() => sa.encodePointCloud(new Float64Array(3)), TypeError);
    assert.throws(() => sa.encodePointCloud(new Float32Array(3), new Uint8Array(3)), TypeError);
    assert.throws(() => sa.decodeGLB(new Uint8Array([1, 2, 3, 4])));
});
