/** Grid spacing, in angstrom, of every position VOLT writes (max rounding error: half of it). */
export const POSITION_STEP: number;

/** RGBA8 (4 per item) or floats in [0, 1] with 3 (RGB) or 4 (RGBA) per item. */
export type Colors = Uint8Array | Float32Array;

export interface Material {
    baseColor?: [number, number, number, number] | number[];
    metallic?: number;
    roughness?: number;
    emissive?: [number, number, number] | number[];
    doubleSided?: boolean;
}

export interface DecodedPrimitive {
    mode: 'points' | 'triangles';
    /** World-space xyz per vertex. */
    positions: Float32Array;
    normals: Float32Array | null;
    /** RGBA8 per vertex. */
    colors: Uint8Array | null;
    indices: Uint32Array | null;
    material: Required<Material>;
}

/**
 * Point cloud as a quantized, meshopt-compressed glTF binary. Points keep their order:
 * point i of the file is positions[i].
 */
export function encodePointCloud(positions: Float32Array, colors?: Colors | null): Buffer;

/** Indexed triangle mesh as a quantized, meshopt-compressed glTF binary. */
export function encodeMesh(
    positions: Float32Array,
    normals: Float32Array,
    indices: Uint32Array | Uint16Array,
    colors?: Colors | null,
    material?: Material
): Buffer;

/** Point and triangle primitives of any glTF binary, in world space. Throws on unusable data. */
export function decodeGLB(bytes: Uint8Array): DecodedPrimitive[];

/** RGBA8 colour of each atom type (types above 7 take the last colour). */
export function colorsByType(types: Uint16Array): Uint8Array;

/** RGBA8 colour of each value on a gradient (see colors.h for the codes) over [min, max]. */
export function propertyColors(values: Float32Array, min: number, max: number, gradient: number): Uint8Array;

/** Taubin smoothing of a triangle mesh, in place. */
export function taubinSmooth(positions: Float32Array, indices: Uint32Array, iterations: number): void;
