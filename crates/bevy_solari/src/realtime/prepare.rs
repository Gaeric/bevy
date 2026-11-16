use super::SolariLighting;
use crate::scene::RaytracingSceneNeedsPreviousFrameData;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_anti_alias::dlss::{
    Dlss, DlssRayReconstructionFeature, ViewDlssRayReconstructionTextures,
};
use bevy_camera::MainPassResolutionOverride;
use bevy_diagnostic::FrameCount;
#[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
use bevy_ecs::query::Has;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    system::{Commands, Query, Res},
};
use bevy_image::ToExtents;
use bevy_math::UVec2;
use bevy_render::{
    camera::ExtractedCamera,
    render_resource::{
        Buffer, BufferDescriptor, BufferInitDescriptor, BufferUsages, TextureDescriptor,
        TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::CachedTexture,
};
use bytemuck::{Pod, Zeroable};
use core::sync::atomic::{AtomicU32, Ordering};

/// Size of the `LightSample` shader struct in bytes.
const LIGHT_SAMPLE_STRUCT_SIZE: u64 = 8;

/// Size of the `ResolvedLightSamplePacked` shader struct in bytes.
const RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE: u64 = 24;

/// Size of the `Reservoir` shader struct in bytes.
const RESERVOIR_STRUCT_SIZE: u64 = 48;

pub const LIGHT_TILE_BLOCKS: u64 = 128;
pub const LIGHT_TILE_SAMPLES_PER_BLOCK: u64 = 1024;

/// Amount of entries in the world cache (must be a power of 2, and >= 2^10)
pub const WORLD_CACHE_SIZE: u64 = 2u64.pow(20);
/// Sum of per-cell field sizes in `WorldCache`. Keep in sync with `realtime_bindings.wgsl`.
const WORLD_CACHE_ENTRY_SIZE: u64 = 84;
/// Size of the fixed `b` array (`array<u32, WORLD_CACHE_SIZE / 1024>`).
const WORLD_CACHE_B_SIZE: u64 = (WORLD_CACHE_SIZE / 1024) * size_of::<u32>() as u64;
/// Offset of `active_cells_count`.
pub const WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET: u64 =
    WORLD_CACHE_SIZE * WORLD_CACHE_ENTRY_SIZE + WORLD_CACHE_B_SIZE;
/// Must stay under wgpu's default `max_storage_buffer_binding_size` (128 MiB or 2^27 bytes).
pub const WORLD_CACHE_BUFFER_SIZE: u64 =
    (WORLD_CACHE_ACTIVE_CELLS_COUNT_OFFSET + size_of::<u32>() as u64).next_multiple_of(16);

/// GPU representation of the user-configurable [`SolariLighting`] settings, plus
/// per-frame state.
///
/// Field order and types must match the `SolariLightingSettings` struct in
/// `realtime_bindings.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SolariLightingUniforms {
    confidence_weight_cap: f32,
    primary_di_samples: u32,
    secondary_di_samples: u32,
    max_bounces: u32,
    world_cache_max_temporal_samples: f32,
    world_cache_direct_light_sample_count: u32,
    world_cache_max_gi_ray_distance: f32,
    world_cache_cell_updates_soft_target: u32,
    world_cache_position_base_cell_size: f32,
    world_cache_position_lod_scale: f32,
    frame_rng: u32,
    reset: u32,
    shadow_terminator_fix: u32,
    shadow_terminator_history_valid: u32,
    _padding: [u32; 2],
}

impl SolariLightingUniforms {
    fn new(
        settings: &SolariLighting,
        frame_count: u32,
        force_reset: bool,
        shadow_terminator_history_valid: bool,
    ) -> Self {
        Self {
            confidence_weight_cap: settings.confidence_weight_cap,
            primary_di_samples: settings.primary_di_samples,
            secondary_di_samples: settings.secondary_di_samples,
            max_bounces: settings.max_bounces,
            world_cache_max_temporal_samples: settings.world_cache_max_temporal_samples,
            world_cache_direct_light_sample_count: settings.world_cache_direct_light_sample_count,
            world_cache_max_gi_ray_distance: settings.world_cache_max_gi_ray_distance,
            world_cache_cell_updates_soft_target: settings.world_cache_cell_updates_soft_target,
            world_cache_position_base_cell_size: settings.world_cache_position_base_cell_size,
            world_cache_position_lod_scale: settings.world_cache_position_lod_scale,
            frame_rng: frame_count.wrapping_mul(5782582),
            reset: (settings.reset || force_reset) as u32,
            shadow_terminator_fix: settings.shadow_terminator_fix as u32,
            shadow_terminator_history_valid: shadow_terminator_history_valid as u32,
            _padding: [0; 2],
        }
    }
}

/// Declares to the raytracing scene whether any view needs last frame's TLAS and light ids.
pub fn setup_raytracing_scene_needs_previous_frame_data(
    views: Query<&SolariLighting>,
    needs_previous_frame_data: Option<Res<RaytracingSceneNeedsPreviousFrameData>>,
    mut commands: Commands,
) {
    let restir_used = views.iter().any(|solari_lighting| solari_lighting.restir);
    match (restir_used, needs_previous_frame_data.is_some()) {
        (true, false) => commands.insert_resource(RaytracingSceneNeedsPreviousFrameData),
        (false, true) => commands.remove_resource::<RaytracingSceneNeedsPreviousFrameData>(),
        _ => {}
    }
}

/// Internal rendering resources used for Solari lighting.
#[derive(Component)]
pub struct SolariLightingResources {
    pub constants: Buffer,
    pub light_tile_samples: Buffer,
    pub light_tile_resolved_samples: Buffer,
    pub reservoirs: Option<SolariReservoirBuffers>,
    pub world_cache: Buffer,
    pub world_cache_active_cells_dispatch: Buffer,
    pub view_size: UVec2,
    pub shadow_terminator_fix: bool,
    pub shadow_terminator_offsets: ShadowTerminatorTextures,
}

/// Fixed current/previous offset textures, owned by this view rather than the texture cache.
/// After successful lighting, copy current into previous and record the rendered frame.
pub struct ShadowTerminatorTextures {
    pub current: CachedTexture,
    pub previous: CachedTexture,
    pub last_rendered_frame: AtomicU32,
}

impl ShadowTerminatorTextures {
    fn new(render_device: &RenderDevice, view_size: UVec2, enabled: bool) -> Self {
        // The enabled pair costs 32 bytes per pixel; disabled views retain only two
        // bind-compatible 1x1 placeholders, never full-resolution offset textures.
        let size = if enabled { view_size } else { UVec2::ONE }.to_extents();
        let texture = |label, usage| {
            let texture = render_device.create_texture(&TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba32Float,
                usage,
                view_formats: &[],
            });
            let default_view = texture.create_view(&TextureViewDescriptor::default());
            CachedTexture {
                texture,
                default_view,
            }
        };
        Self {
            current: texture(
                "solari_lighting_shadow_terminator_offsets_current",
                TextureUsages::STORAGE_BINDING
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
            ),
            previous: texture(
                "solari_lighting_shadow_terminator_offsets_previous",
                TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            ),
            last_rendered_frame: AtomicU32::new(u32::MAX),
        }
    }
}

/// History is usable only when this exact resource set was rendered last frame.
fn shadow_terminator_history_valid(
    enabled: bool,
    reset: bool,
    mode_changed: bool,
    resources_reusable: bool,
    last_rendered_frame: u32,
    frame_count: u32,
) -> bool {
    enabled
        && !reset
        && !mode_changed
        && resources_reusable
        // Frame zero must not mistake the invalid sentinel for the preceding frame.
        && frame_count != 0
        && last_rendered_frame == frame_count.wrapping_sub(1)
}

pub struct SolariReservoirBuffers {
    pub a: Buffer,
    pub b: Buffer,
}

pub fn prepare_solari_lighting_resources(
    #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))] mut query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&mut SolariLightingResources>,
        Option<&MainPassResolutionOverride>,
    )>,
    #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))] mut query: Query<(
        Entity,
        &ExtractedCamera,
        &SolariLighting,
        Option<&mut SolariLightingResources>,
        Option<&MainPassResolutionOverride>,
        Has<Dlss<DlssRayReconstructionFeature>>,
    )>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
    mut commands: Commands,
) {
    for query_item in &mut query {
        #[cfg(any(not(feature = "dlss"), feature = "force_disable_dlss"))]
        let (entity, camera, solari_lighting, solari_lighting_resources, resolution_override) =
            query_item;
        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        let (
            entity,
            camera,
            solari_lighting,
            solari_lighting_resources,
            resolution_override,
            has_dlss_rr,
        ) = query_item;

        let Some(mut view_size) = camera.physical_viewport_size else {
            continue;
        };
        if let Some(MainPassResolutionOverride(resolution_override)) = resolution_override {
            view_size = *resolution_override;
        }

        let mut reusable = solari_lighting_resources.filter(|r| {
            r.view_size == view_size && r.reservoirs.is_some() == solari_lighting.restir
        });
        let mode_changed = reusable
            .as_ref()
            .is_some_and(|r| r.shadow_terminator_fix != solari_lighting.shadow_terminator_fix);
        let force_reset = reusable.is_none() || mode_changed;
        if let Some(resources) = reusable.as_mut() {
            if mode_changed {
                // Keep the world cache, reservoirs, and light tiles across toggles.
                // Replacing these owned handles releases the old offset allocation.
                resources.shadow_terminator_offsets = ShadowTerminatorTextures::new(
                    &render_device,
                    view_size,
                    solari_lighting.shadow_terminator_fix,
                );
            }
            if solari_lighting.reset || force_reset {
                // Invalidate before uploading uniforms, even if lighting is skipped.
                resources
                    .shadow_terminator_offsets
                    .last_rendered_frame
                    .store(u32::MAX, Ordering::Relaxed);
            }
        }
        let history_valid = shadow_terminator_history_valid(
            solari_lighting.shadow_terminator_fix,
            solari_lighting.reset || force_reset,
            mode_changed,
            reusable.is_some(),
            reusable.as_ref().map_or(u32::MAX, |resources| {
                resources
                    .shadow_terminator_offsets
                    .last_rendered_frame
                    .load(Ordering::Relaxed)
            }),
            frame_count.0,
        );
        let uniforms =
            SolariLightingUniforms::new(solari_lighting, frame_count.0, force_reset, history_valid);

        if let Some(mut solari_lighting_resources) = reusable {
            solari_lighting_resources.shadow_terminator_fix = solari_lighting.shadow_terminator_fix;
            // The constants uniform can change every frame, so always upload it.
            render_queue.write_buffer(
                &solari_lighting_resources.constants,
                0,
                bytemuck::bytes_of(&uniforms),
            );
            continue;
        }

        let constants = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("solari_lighting_constants"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let light_tile_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_samples"),
            size: LIGHT_TILE_BLOCKS * LIGHT_TILE_SAMPLES_PER_BLOCK * LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let light_tile_resolved_samples = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_light_tile_resolved_samples"),
            size: LIGHT_TILE_BLOCKS
                * LIGHT_TILE_SAMPLES_PER_BLOCK
                * RESOLVED_LIGHT_SAMPLE_STRUCT_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let reservoirs = solari_lighting.restir.then(|| {
            let reservoirs_buffer = |name| {
                render_device.create_buffer(&BufferDescriptor {
                    label: Some(name),
                    size: (view_size.x * view_size.y) as u64 * RESERVOIR_STRUCT_SIZE,
                    usage: BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            SolariReservoirBuffers {
                a: reservoirs_buffer("solari_lighting_reservoirs_a"),
                b: reservoirs_buffer("solari_lighting_reservoirs_b"),
            }
        });

        let world_cache = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache"),
            size: WORLD_CACHE_BUFFER_SIZE,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let world_cache_active_cells_dispatch = render_device.create_buffer(&BufferDescriptor {
            label: Some("solari_lighting_world_cache_active_cells_dispatch"),
            size: size_of::<[u32; 3]>() as u64,
            usage: BufferUsages::INDIRECT | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        commands.entity(entity).insert(SolariLightingResources {
            constants,
            light_tile_samples,
            light_tile_resolved_samples,
            reservoirs,
            world_cache,
            world_cache_active_cells_dispatch,
            view_size,
            shadow_terminator_fix: solari_lighting.shadow_terminator_fix,
            shadow_terminator_offsets: ShadowTerminatorTextures::new(
                &render_device,
                view_size,
                solari_lighting.shadow_terminator_fix,
            ),
        });

        #[cfg(all(feature = "dlss", not(feature = "force_disable_dlss")))]
        if has_dlss_rr {
            let diffuse_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_diffuse_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let diffuse_albedo_view = diffuse_albedo.create_view(&TextureViewDescriptor::default());

            let specular_albedo = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_albedo"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_albedo_view =
                specular_albedo.create_view(&TextureViewDescriptor::default());

            let normal_roughness = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_normal_roughness"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let normal_roughness_view =
                normal_roughness.create_view(&TextureViewDescriptor::default());

            let dlss_rr_depth = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_dlss_rr_depth"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let dlss_rr_depth_view = dlss_rr_depth.create_view(&TextureViewDescriptor::default());

            let specular_motion_vectors = render_device.create_texture(&TextureDescriptor {
                label: Some("solari_lighting_specular_motion_vectors"),
                size: view_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            let specular_motion_vectors_view =
                specular_motion_vectors.create_view(&TextureViewDescriptor::default());

            commands
                .entity(entity)
                .insert(ViewDlssRayReconstructionTextures {
                    diffuse_albedo: CachedTexture {
                        texture: diffuse_albedo,
                        default_view: diffuse_albedo_view,
                    },
                    specular_albedo: CachedTexture {
                        texture: specular_albedo,
                        default_view: specular_albedo_view,
                    },
                    normal_roughness: CachedTexture {
                        texture: normal_roughness,
                        default_view: normal_roughness_view,
                    },
                    depth: CachedTexture {
                        texture: dlss_rr_depth,
                        default_view: dlss_rr_depth_view,
                    },
                    specular_motion_vectors: CachedTexture {
                        texture: specular_motion_vectors,
                        default_view: specular_motion_vectors_view,
                    },
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_terminator_uniform_layout_and_defaults() {
        // Matches sixteen scalar fields in realtime/bindings.wesl (64 bytes).
        assert_eq!(size_of::<SolariLightingUniforms>(), 64);
        assert_eq!(
            core::mem::offset_of!(SolariLightingUniforms, shadow_terminator_fix),
            48
        );
        assert_eq!(
            core::mem::offset_of!(SolariLightingUniforms, shadow_terminator_history_valid),
            52
        );
        let mut settings = SolariLighting::default();
        settings.reset = false;
        let uniforms = SolariLightingUniforms::new(&settings, 0, false, false);
        assert_eq!(uniforms.shadow_terminator_fix, 0);
        assert_eq!(uniforms.reset, 0);
        settings.shadow_terminator_fix = true;
        let uniforms = SolariLightingUniforms::new(&settings, 0, true, false);
        assert_eq!(uniforms.shadow_terminator_fix, 1);
        assert_eq!(uniforms.reset, 1);
        assert_eq!(uniforms.shadow_terminator_history_valid, 0);
        assert_eq!(uniforms._padding, [0; 2]);
        let uniforms = SolariLightingUniforms::new(&settings, 1, false, true);
        assert_eq!(uniforms.shadow_terminator_history_valid, 1);
        assert_eq!(uniforms.reset, 0);
    }

    #[test]
    fn shadow_terminator_history_requires_consecutive_successful_frames() {
        let valid =
            |last, frame| shadow_terminator_history_valid(true, false, false, true, last, frame);
        assert!(valid(0, 1));
        assert!(valid(41, 42));
        assert!(valid(u32::MAX - 1, u32::MAX));
        assert!(!valid(u32::MAX, 0)); // Initial sentinel and frame-count wrap.
        assert!(!valid(u32::MAX, 1)); // New or invalidated textures.
        assert!(!valid(40, 42)); // A skipped lighting frame.
        assert!(!valid(42, 42)); // Already rendered this frame, not last frame.
    }

    #[test]
    fn shadow_terminator_history_rejects_reset_toggle_and_reallocation() {
        assert!(shadow_terminator_history_valid(
            true, false, false, true, 6, 7
        ));
        assert!(!shadow_terminator_history_valid(
            false, false, false, true, 6, 7
        ));
        assert!(!shadow_terminator_history_valid(
            true, true, false, true, 6, 7
        ));
        assert!(!shadow_terminator_history_valid(
            true, false, true, true, 6, 7
        ));
        // A resize or ReSTIR mode change creates a new resource set.
        assert!(!shadow_terminator_history_valid(
            true, false, false, false, 6, 7
        ));
        // Reset invalidates the stamp even when that frame never renders.
        let last_rendered_frame = AtomicU32::new(6);
        last_rendered_frame.store(u32::MAX, Ordering::Relaxed);
        assert!(!shadow_terminator_history_valid(
            true,
            false,
            false,
            true,
            last_rendered_frame.load(Ordering::Relaxed),
            7,
        ));
        // A successful render after a toggle/reset seeds the next frame's history.
        last_rendered_frame.store(7, Ordering::Relaxed);
        assert!(shadow_terminator_history_valid(
            true,
            false,
            false,
            true,
            last_rendered_frame.load(Ordering::Relaxed),
            8,
        ));
    }
}
