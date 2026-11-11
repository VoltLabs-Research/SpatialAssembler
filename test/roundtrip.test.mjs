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

// A plain float32 glTF binary, as VOLT wrote models before this format and other tools still do.
const float32Glb = ({ attributes, indices, mode, node, material }) => {
    const views = [];
    const accessors = [];
    const chunks = [];
    let offset = 0;
    const addView = (array, componentType, type, target) => {
        views.push({ buffer: 0, byteOffset: offset, byteLength: array.byteLength, target });
        accessors.push({ bufferView: views.length - 1, componentType, count: array.length / { SCALAR: 1, VEC3: 3 }[type], type });
        chunks.push(Buffer.from(array.buffer, array.byteOffset, array.byteLength));
        offset += array.byteLength;
        return accessors.length - 1;
    };
    const primitive = { mode, attributes: {} };
    for (const [name, array] of Object.entries(attributes)) {
        primitive.attributes[name] = addView(array, 5126, 'VEC3', 34962);
    }
    if (indices) primitive.indices = addView(indices, 5125, 'SCALAR', 34963);
    if (material) primitive.material = 0;
    accessors[primitive.attributes.POSITION].min = [0, 0, 0];
    accessors[primitive.attributes.POSITION].max = [1, 1, 1];

    const json = Buffer.from(JSON.stringify({
        asset: { version: '2.0' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0, ...node }],
        meshes: [{ primitives: [primitive] }],
        ...(material ? { materials: [material] } : {}),
        buffers: [{ byteLength: offset }],
        bufferViews: views,
        accessors
    }));
    const jsonChunk = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 0x20)]);
    const bin = Buffer.concat(chunks);
    const header = Buffer.alloc(12);
    header.writeUInt32LE(0x46546c67, 0);
    header.writeUInt32LE(2, 4);
    header.writeUInt32LE(12 + 8 + jsonChunk.length + 8 + bin.length, 8);
    const chunkHeader = (length, type) => {
        const out = Buffer.alloc(8);
        out.writeUInt32LE(length, 0);
        out.writeUInt32LE(type, 4);
        return out;
    };
    return Buffer.concat([header, chunkHeader(jsonChunk.length, 0x4e4f534a), jsonChunk, chunkHeader(bin.length, 0x004e4942), bin]);
};

test('float32 point clouds decode in world coordinates with RGBA8 colours', () => {
    const positions = new Float32Array([0, 0, 0, 1.25, -2.5, 3.75]);
    const glb = float32Glb({
        mode: 0,
        attributes: { POSITION: positions, COLOR_0: new Float32Array([1, 0, 0, 0, 0.5, 1]) },
        node: { translation: [10, 20, 30] }
    });
    const [points] = sa.decodeGLB(glb);
    assert.equal(points.mode, 'points');
    assert.deepEqual([...points.positions], [10, 20, 30, 11.25, 17.5, 33.75]);
    assert.deepEqual([...points.colors], [255, 0, 0, 255, 0, 128, 255, 255]);
});

test('float32 meshes decode with their indices, normals and material', () => {
    const glb = float32Glb({
        mode: 4,
        attributes: {
            POSITION: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
            NORMAL: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
        },
        indices: new Uint32Array([0, 1, 2, 2, 1, 3]),
        material: { pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.6, 1], metallicFactor: 0.3, roughnessFactor: 0.7 } }
    });
    const [mesh] = sa.decodeGLB(glb);
    assert.equal(mesh.mode, 'triangles');
    assert.deepEqual([...mesh.indices], [0, 1, 2, 2, 1, 3]);
    assert.deepEqual([...mesh.normals.subarray(0, 3)], [0, 0, 1]);
    assert.ok(Math.abs(mesh.material.baseColor[1] - 0.4) < 1e-6 && Math.abs(mesh.material.roughness - 0.7) < 1e-6);

    const [reencoded] = sa.decodeGLB(sa.encodeMesh(mesh.positions, mesh.normals, mesh.indices, mesh.colors, mesh.material));
    assert.deepEqual([...reencoded.indices], [0, 1, 2, 2, 1, 3]);
    assert.ok(Math.abs(reencoded.positions[9] - 1) <= sa.POSITION_STEP / 2);
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
