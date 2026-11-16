//! Experimental Ray Tracing Gems II, chapter 4 shading-point offset.
//! Recover the primary triangle with an extra camera ray instead of changing the G-buffer.
import bevy_render::maths::affine3_to_square;
import package::realtime::bindings::{constants, view};
import package::scene::bindings::{trace_ray, transforms, geometry_ids, load_vertices, transform_positions, RAY_T_MIN};

enable wgpu_ray_query;

/// Only changes the origin of primary NEE shadow rays, not BRDF evaluation, ray directions
/// used by the BRDF, indirect reconnections, or the G-buffer position. Returns zero on a
/// raster/raytracing mismatch. This prototype supports perspective cameras only.
fn primary_shadow_terminator_offset(world_position: vec3<f32>, camera_position: vec3<f32>) -> vec3<f32> {
    if constants.shadow_terminator_fix == 0u || view.clip_from_view[3].w == 1.0 { return vec3(0.0); }

    let camera_ray = world_position - camera_position;
    let camera_distance = length(camera_ray);
    if camera_distance <= RAY_T_MIN { return vec3(0.0); }

    // Permit depth reconstruction roundoff, but reject a different foreground surface.
    // This is a prototype heuristic: nearly coincident raster-only/ray-only surfaces can
    // still be ambiguous. Do not search past an occluder to find a more convenient hit.
    let tolerance = max(4.0 * RAY_T_MIN, camera_distance * 0.0001);
    let hit = trace_ray(camera_position, camera_ray / camera_distance, 0.0, camera_distance + tolerance, RAY_FLAG_NONE);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE || abs(hit.t - camera_distance) > tolerance {
        return vec3(0.0);
    }

    let vertices = load_vertices(geometry_ids[hit.instance_custom_data], hit.primitive_index);
    let transform = affine3_to_square(transforms[hit.instance_custom_data]);
    let positions = transform_positions(transform, vertices);
    let bary = vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics);
    let p = mat3x3(positions[0], positions[1], positions[2]) * bary;
    if distance(p, world_position) > tolerance { return vec3(0.0); }

    // Inverse-transpose normal transformation, including nonuniform and negative scale.
    // Deliberately use vertex normals, never the normal-map-perturbed G-buffer normal.
    let linear = mat3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
    let cofactor = mat3x3(cross(linear[1], linear[2]), cross(linear[2], linear[0]), cross(linear[0], linear[1]));
    let determinant = dot(linear[0], cofactor[0]);
    if abs(determinant) < 1e-20 { return vec3(0.0); }
    let normal_transform = sign(determinant) * cofactor;
    var offset = vec3(0.0);
    for (var i = 0u; i < 3u; i++) {
        let n_raw = normal_transform * vertices[i].normal;
        if dot(n_raw, n_raw) < 1e-20 { return vec3(0.0); }
        let n = normalize(n_raw);
        offset -= bary[i] * min(0.0, dot(p - positions[i], n)) * n;
    }
    return offset;
}
