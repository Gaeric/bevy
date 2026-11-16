//! Ray Tracing Gems II chapter 4: produce a per-pixel world-space shadow-origin offset.
//! Lighting only samples the result; triangle recovery is performed once per pixel here.
//! This producer can later be replaced by a raster/visibility-buffer implementation.
import bevy_render::maths::affine3_to_square;
import bevy_render::view::View;
import package::realtime::bindings::SolariLightingSettings;
import package::realtime::gbuffer_utils::reconstruct_world_position;
import package::scene::bindings::{trace_ray, transforms, geometry_ids, load_vertices, transform_positions, RAY_T_MIN};

enable wgpu_ray_query;

@group(1) @binding(0) var depth_buffer: texture_depth_2d;
@group(1) @binding(1) var<uniform> view: View;
@group(1) @binding(2) var<uniform> constants: SolariLightingSettings;
// xyz = world-space displacement (not an absolute position); w = valid matched triangle.
@group(1) @binding(3) var offsets: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn generate_shadow_terminator_offsets(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id.xy >= textureDimensions(offsets)) { return; }
    // Every texel is overwritten, including background and failed matches, so stale
    // offsets can never survive disocclusions. Host skips this dispatch when disabled.
    var result = vec4(0.0);
    let depth = textureLoad(depth_buffer, id.xy, 0);
    if constants.shadow_terminator_fix != 0u && depth != 0.0 {
        let p = reconstruct_world_position(id.xy, depth, view.main_pass_viewport.zw, view.world_from_clip);
        // Start at the raster near plane (rather than the camera), respecting clipped
        // foreground geometry and supporting perspective and orthographic projections.
        let ray_origin = reconstruct_world_position(id.xy, 1.0, view.main_pass_viewport.zw, view.world_from_clip);
        result = recover_shadow_terminator_offset(p, ray_origin);
    }
    textureStore(offsets, id.xy, result);
}

fn recover_shadow_terminator_offset(world_position: vec3<f32>, ray_origin: vec3<f32>) -> vec4<f32> {
    let camera_ray = world_position - ray_origin;
    let camera_distance = length(camera_ray);
    if !(camera_distance > 0.0) { return vec4(0.0); }

    // Heuristic position match, not a guaranteed raster/ray geometry identity check.
    // Do not search past a raytracing occluder to find a more convenient triangle.
    let tolerance = max(4.0 * RAY_T_MIN, camera_distance * 0.0001);
    let hit = trace_ray(ray_origin, camera_ray / camera_distance, 0.0, camera_distance + tolerance, RAY_FLAG_NONE);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE || abs(hit.t - camera_distance) > tolerance {
        return vec4(0.0);
    }

    let vertices = load_vertices(geometry_ids[hit.instance_custom_data], hit.primitive_index);
    let transform = affine3_to_square(transforms[hit.instance_custom_data]);
    let positions = transform_positions(transform, vertices);
    let bary = vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    let p = mat3x3(positions[0], positions[1], positions[2]) * bary;
    if distance(p, world_position) > tolerance { return vec4(0.0); }

    // Inverse-transpose normal transformation, including nonuniform and negative scale.
    // Use each vertex normal, never the normal-map-perturbed G-buffer normal.
    let linear = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let cofactor = mat3x3(cross(linear[1], linear[2]), cross(linear[2], linear[0]), cross(linear[0], linear[1]));
    let determinant = dot(linear[0], cofactor[0]);
    if !(abs(determinant) > 1e-20) { return vec4(0.0); }
    let normal_transform = sign(determinant) * cofactor;
    var offset = vec3(0.0);
    for (var i = 0u; i < 3u; i++) {
        let n_raw = normal_transform * vertices[i].normal;
        if !(dot(n_raw, n_raw) > 1e-20) { return vec4(0.0); }
        let n = normalize(n_raw);
        offset -= bary[i] * min(0.0, dot(p - positions[i], n)) * n;
    }
    // Reject NaN/infinity from malformed mesh data before writing reusable history.
    if !all(abs(offset) <= vec3(3.402823e38)) { return vec4(0.0); }
    return vec4(offset, 1.0);
}
