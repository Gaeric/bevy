//! AMD FidelityFX Super Resolution 3 (FSR3).
//!
//! FSR3 is a temporal upscaling and anti-aliasing technique that uses
//! machine learning-based upscaling to render at a lower resolution
//! and upscale to the target resolution.
//!
//! # Usage
//! 1. Add the `Fsr3` component to your camera entity
//! 2. Optionally set a specific `Fsr3QualityMode` (defaults to `Quality`)
//! 3. The camera must have HDR enabled
//! 4. Optionally adjust sharpening settings
//!
//! # Example
//! ```ignore
//! commands.spawn((
//!     Camera3d::default(),
//!     Fsr3 {
//!         quality_mode: Fsr3QualityMode::Quality,
//!         enable_sharpening: true,
//!         sharpness: 0.8,
//!         ..default()
//!     },
//! ));
//! ```

mod node;

use bevy_app::{App, Plugin};
use bevy_camera::Hdr;
use bevy_camera::{
    Camera, Camera3d, CameraMainTextureUsages, MainPassResolutionOverride, Projection,
};
use bevy_core_pipeline::{
    prepass::{DepthPrepass, MotionVectorPrepass},
    schedule::Core3d,
    Core3dSystems,
};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::With,
    reflect::ReflectComponent,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res, ResMut},
};
use bevy_image::ToExtents;
use bevy_math::{UVec2, Vec2, Vec4Swizzles};
use bevy_reflect::{std_traits::ReflectDefault, Reflect, ReflectFromReflect};
use bevy_render::sync_component::SyncComponent;
use bevy_render::{
    camera::{ExtractedCamera, MipBias, TemporalJitter},
    render_resource::{
        Buffer, BufferDescriptor, BufferUsages, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_component::SyncComponentPlugin,
    sync_world::RenderEntity,
    texture::{CachedTexture, TextureCache},
    view::ExtractedView,
    ExtractSchedule, MainWorld, Render, RenderApp, RenderSystems,
};
use std::{mem::size_of, sync::Mutex};
use tracing::info;
use wgpu_ffx::{FsrContext, FsrContextFlags, FsrContextInfo};

/// Plugin for AMD FidelityFX Super Resolution 3.
///
/// See [`Fsr3`] for more details.
#[derive(Default)]
pub struct Fsr3Plugin;

impl Plugin for Fsr3Plugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(SyncComponentPlugin::<Fsr3>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .add_systems(ExtractSchedule, extract_fsr3_settings)
            .add_systems(
                Render,
                (
                    prepare_fsr3_jitter_and_context.in_set(RenderSystems::CreateViews),
                    prepare_fsr3_textures.in_set(RenderSystems::PrepareResources),
                ),
            );

        render_app.add_systems(
            Core3d,
            node::fsr_super_resolution.in_set(Core3dSystems::EarlyPostProcess),
        );
    }
}

/// Component to apply FSR3 temporal upscaling and anti-aliasing to a 3D camera.
///
/// FSR3 is AMD's temporal upscaling solution that renders at a lower resolution
/// and upscales to the target resolution using temporal data.
///
/// # Tradeoffs
///
/// Pros:
/// * Much better performance by rendering at lower resolution
/// * High quality temporal anti-aliasing
/// * Works on AMD, NVIDIA, and Intel GPUs
/// * Includes optional sharpening pass
///
/// Cons:
/// * Requires HDR rendering
/// * May exhibit ghosting artifacts with fast motion
/// * Requires accurate motion vectors
///
/// # Usage Notes
///
/// Any camera with this component must have HDR enabled (the `Hdr` component).
/// The camera must also disable [`Msaa`] by setting it to [`Msaa::Off`].
///
/// FSR3 requires accurate motion vectors for everything on screen. Custom
/// rendering code must write proper motion vectors to work correctly with FSR3.
#[derive(Component, Reflect, Clone)]
#[reflect(Component, Default, FromReflect)]
#[require(TemporalJitter, MipBias, DepthPrepass, MotionVectorPrepass, Hdr)]
pub struct Fsr3 {
    /// Quality mode controlling the render resolution and upscaling ratio.
    #[reflect(default)]
    pub quality_mode: Fsr3QualityMode,

    /// Set to true to delete the saved temporal history (past frames).
    ///
    /// Useful for preventing ghosting when the history is no longer
    /// representative of the current frame, such as in sudden camera cuts.
    ///
    /// After setting this to true, it will automatically be toggled
    /// back to false at the end of the frame.
    pub reset: bool,

    /// Enable the sharpening pass.
    pub enable_sharpening: bool,

    /// Sharpening strength (0.0 = no sharpening, 1.0 = maximum sharpening).
    pub sharpness: f32,
}

impl Default for Fsr3 {
    fn default() -> Self {
        Self {
            quality_mode: Fsr3QualityMode::Quality,
            reset: true,
            enable_sharpening: true,
            sharpness: 0.8,
        }
    }
}

impl SyncComponent for Fsr3 {
    type Target = Self;
}

/// Quality modes for FSR3, controlling the render resolution and upscaling ratio.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Fsr3QualityMode {
    /// Native rendering with no upscaling (1.0x).
    /// Provides maximum quality but no performance benefit.
    NativeAA,

    /// Quality mode with 1.5x upscaling.
    /// Best balance of quality and performance for most use cases.
    #[default]
    Quality,

    /// Balanced mode with 1.7x upscaling.
    /// Good quality with better performance than Quality mode.
    Balanced,

    /// Performance mode with 2.0x upscaling.
    /// Significant performance improvement with acceptable quality loss.
    Performance,

    /// Ultra Performance mode with 3.0x upscaling.
    /// Maximum performance improvement with most quality loss.
    UltraPerformance,
}

impl Fsr3QualityMode {
    /// Get the upscaling ratio for this quality mode.
    pub fn scale_factor(self) -> f32 {
        match self {
            Fsr3QualityMode::NativeAA => 1.0,
            Fsr3QualityMode::Quality => 1.5,
            Fsr3QualityMode::Balanced => 1.7,
            Fsr3QualityMode::Performance => 2.0,
            Fsr3QualityMode::UltraPerformance => 3.0,
        }
    }

    /// Calculate the render resolution from the target (upscaled) resolution.
    pub fn render_resolution(self, upscale_resolution: UVec2) -> UVec2 {
        let scale = self.scale_factor();
        UVec2::new(
            (upscale_resolution.x as f32 / scale).round() as u32,
            (upscale_resolution.y as f32 / scale).round() as u32,
        )
    }
}

/// Render context for FSR3, stored per camera.
#[derive(Component)]
struct Fsr3RenderContext {
    context: Mutex<FsrContext>,
    quality_mode: Fsr3QualityMode,
    max_render_size: [u32; 2],
    max_upscale_size: [u32; 2],
}

/// Textures needed for FSR3 rendering.
#[derive(Component)]
struct Fsr3Textures {
    dilated_depth: CachedTexture,
    dilated_motion_vectors: CachedTexture,
    reconstructed_previous_depth: Buffer,
}

fn extract_fsr3_settings(mut commands: Commands, mut main_world: ResMut<MainWorld>) {
    let mut cameras_3d =
        main_world.query::<(RenderEntity, &Camera, &Projection, Option<&mut Fsr3>)>();

    for (entity, camera, projection, fsr3_settings) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");

        if let Some(mut fsr3_settings) = fsr3_settings
            && camera.is_active
            && projection.is_perspective()
        {
            entity_commands.insert(fsr3_settings.clone());
            fsr3_settings.reset = false;
        } else {
            entity_commands.remove::<(
                Fsr3,
                Fsr3RenderContext,
                Fsr3Textures,
                MainPassResolutionOverride,
            )>();
        }
    }
}

fn prepare_fsr3_jitter_and_context(
    mut commands: Commands,
    mut query: Query<
        (
            Entity,
            &ExtractedView,
            &Fsr3,
            &mut Camera3d,
            &mut CameraMainTextureUsages,
            &mut TemporalJitter,
            &mut MipBias,
            Option<&mut Fsr3RenderContext>,
        ),
        (
            With<Camera3d>,
            With<TemporalJitter>,
            With<DepthPrepass>,
            With<MotionVectorPrepass>,
        ),
    >,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    frame_count: Res<FrameCount>,
) {
    for (
        entity,
        view,
        fsr3,
        mut camera_3d,
        mut camera_main_texture_usages,
        mut temporal_jitter,
        mut mip_bias,
        fsr3_context,
    ) in &mut query
    {
        // Ensure textures have correct usage flags
        camera_main_texture_usages.0 |= TextureUsages::STORAGE_BINDING;

        let mut depth_texture_usages = TextureUsages::from(camera_3d.depth_texture_usages);
        depth_texture_usages |= TextureUsages::TEXTURE_BINDING;
        camera_3d.depth_texture_usages = depth_texture_usages.into();

        let upscale_resolution = view.viewport.zw();
        let render_resolution = fsr3.quality_mode.render_resolution(upscale_resolution);

        // Calculate jitter using FSR3's jitter generation
        let phase_count = wgpu_ffx::get_jitter_phase_count(
            render_resolution.x as i32,
            upscale_resolution.x as i32,
        );
        let jitter = wgpu_ffx::get_jitter_offset(frame_count.0 as i32, phase_count);
        temporal_jitter.offset = Vec2::from(jitter);

        // Calculate mip bias
        let scale_factor = fsr3.quality_mode.scale_factor();
        mip_bias.0 = -(scale_factor.log2());

        // Check if we need to create or update the FSR3 context
        let needs_new_context = match fsr3_context {
            Some(ctx) => {
                ctx.quality_mode != fsr3.quality_mode
                    || ctx.max_upscale_size != [upscale_resolution.x, upscale_resolution.y]
            }
            None => true,
        };

        if needs_new_context {
            // Calculate maximum render size based on quality mode
            let max_render_size = fsr3.quality_mode.render_resolution(upscale_resolution);

            // Setup FSR3 context flags
            let mut flags = FsrContextFlags::HIGH_DYNAMIC_RANGE
                | FsrContextFlags::DEPTH_INVERTED
                | FsrContextFlags::DEPTH_INFINITE;

            // Check if using infinite depth by examining the projection matrix
            // For infinite far plane with reversed z: clip_from_view[3][2] == near and far == infinity
            // We check if w_axis.w == 0.0 (perspective) and assume infinite if so
            let clip_from_view = view.clip_from_view;
            if clip_from_view.w_axis.w == 0.0 {
                // Perspective projection - assume infinite depth for now
                // (Bevy typically uses infinite reversed-z for perspective)
                flags |= FsrContextFlags::DEPTH_INFINITE;
            }

            info!("max_render_size: {max_render_size:?}, max_upscale_size: {upscale_resolution:?}");
            info!("fsr flags: {flags:?}");

            // Create FSR3 context
            let context_info = FsrContextInfo {
                device: render_device.wgpu_device().clone(),
                // queue: wgpu::Queue::clone(&render_queue),
                // max_render_size: [max_render_size.x, max_render_size.y],
                // max_upscale_size: [upscale_resolution.x, upscale_resolution.y],
                flags,
            };

            let context = FsrContext::new(context_info);

            commands.entity(entity).insert((
                Fsr3RenderContext {
                    context: Mutex::new(context),
                    quality_mode: fsr3.quality_mode,
                    max_render_size: [max_render_size.x, max_render_size.y],
                    max_upscale_size: [upscale_resolution.x, upscale_resolution.y],
                },
                MainPassResolutionOverride(render_resolution),
            ));
        } else {
            // Update resolution override in case it changed
            commands
                .entity(entity)
                .insert(MainPassResolutionOverride(render_resolution));
        }
    }
}

fn prepare_fsr3_textures(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    render_device: Res<RenderDevice>,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &Fsr3), With<Fsr3>>,
) {
    for (entity, camera, view, fsr3) in &views {
        let upscale_size = view.viewport.zw();

        // Create dilated depth texture (output texture, not used after FSR3)
        let dilated_depth = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("fsr3_dilated_depth"),
                size: upscale_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R32Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Create dilated motion vectors texture (output texture, not used after FSR3)
        let dilated_motion_vectors = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("fsr3_dilated_motion_vectors"),
                size: upscale_size.to_extents(),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg16Float,
                usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        );

        // Create reconstructed previous depth buffer
        // Size needs to match the expected buffer size for FSR3
        let buffer_size = (upscale_size.x * upscale_size.y * size_of::<f32>() as u32) as u64;
        let reconstructed_previous_depth = render_device.create_buffer(&BufferDescriptor {
            label: Some("fsr3_reconstructed_previous_depth"),
            size: buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        commands.entity(entity).insert(Fsr3Textures {
            dilated_depth,
            dilated_motion_vectors,
            reconstructed_previous_depth,
        });
    }
}
