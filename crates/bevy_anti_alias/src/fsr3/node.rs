use bevy_camera::MainPassResolutionOverride;
use bevy_core_pipeline::prepass::ViewPrepassTextures;
use bevy_ecs::system::{Res, ResMut};
use bevy_math::Vec4Swizzles;
use bevy_render::{
    camera::{ExtractedCamera, TemporalJitter},
    renderer::{RenderContext, RenderQueue, ViewQuery},
    view::{ExtractedView, Msaa, ViewTarget},
};
use bevy_time::Time;
use tracing::warn;
use wgpu_ffx::{FsrDispatchFlags, FsrDispatchInfo};

use crate::fsr3::{Fsr3, Fsr3RenderContext, Fsr3Textures};

pub fn fsr_super_resolution(
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewTarget,
        &ViewPrepassTextures,
        &Fsr3,
        &Fsr3RenderContext,
        &Fsr3Textures,
        &TemporalJitter,
        &MainPassResolutionOverride,
        &Msaa,
    )>,
    mut render_context: RenderContext,
    render_queue: Res<RenderQueue>,
    time: Res<Time>,
) {
    let (
        camera,
        view,
        view_target,
        prepass_textures,
        fsr3,
        fsr3_context,
        fsr3_textures,
        temporal_jitter,
        resolution_override,
        msaa,
    ) = view.into_inner();

    if *msaa != Msaa::Off {
        warn!("FSR3 requires MSAA to be disabled");
        return;
    }

    let (Some(prepass_motion_vectors_texture), Some(prepass_depth_texture)) =
        (&prepass_textures.motion_vectors, &prepass_textures.depth)
    else {
        return;
    };

    let view_target = view_target.post_process_write();

    let upscale_size = view.viewport.zw();
    let render_size = resolution_override.0;

    // Extract camera parameters from the projection matrix
    // For perspective projection (infinite reverse z):
    // clip_from_view[3][3] == 0.0 indicates perspective
    // near plane is at clip_from_view[3][2]
    // fov can be derived from clip_from_view[1][1] (which is f = 1/tan(fov/2))
    let clip_from_view = view.clip_from_view;

    let (camera_fov_y, camera_near, camera_far) = if clip_from_view.w_axis.w == 0.0 {
        // Perspective projection
        let f = clip_from_view.y_axis.y;
        let fov_y = 2.0 * (1.0 / f).atan();
        let near = clip_from_view.w_axis.z;
        // For infinite far plane (reversed z), far is f32::INFINITY
        let far = f32::INFINITY;
        (fov_y, far, near)
    } else {
        warn!("FSR3 requires a perspective camera projection");
        return;
    };

    // Calculate motion vector scale
    // Bevy's motion vectors are in render resolution, FSR3 expects them in pixels
    let motion_vector_scale = [-(render_size.x as f32), -(render_size.y as f32)];

    let max_render_size = fsr3_context.max_render_size;
    let max_upscale_size = fsr3_context.max_upscale_size;

    let context = fsr3_context.context.lock().unwrap();

    // Create a command encoder specifically for FSR3
    let encoder = render_context.command_encoder();

    // Build dispatch info - wgpu_ffx expects raw wgpu types
    let mut dispatch_info = FsrDispatchInfo {
        color: wgpu::Texture::clone(&view_target.source_texture),
        depth: wgpu::Texture::clone(&prepass_depth_texture.texture.texture),
        motion_vectors: wgpu::Texture::clone(&prepass_motion_vectors_texture.texture.texture),
        dilated_depth: wgpu::Texture::clone(&fsr3_textures.dilated_depth.texture),
        dilated_motion_vectors: wgpu::Texture::clone(&fsr3_textures.dilated_motion_vectors.texture),
        reconstructed_previous_depth: wgpu::Buffer::clone(
            &fsr3_textures.reconstructed_previous_depth,
        ),
        output: wgpu::Texture::clone(&view_target.destination_texture),
        render_size: [render_size.x, render_size.y],
        upscale_size: [upscale_size.x, upscale_size.y],
        jitter_offset: [temporal_jitter.offset.x, temporal_jitter.offset.y],
        motion_vector_scale,
        camera_fov_y,
        camera_near,
        camera_far,
        frame_time_delta: time.delta_secs() * 1000.0,
        reset_history: fsr3.reset,
        enable_sharpening: fsr3.enable_sharpening,
        sharpness: fsr3.sharpness,
        pre_exposure: 1.0, // TODO: integrate with auto-exposure
        view_space_to_meters_factor: 1.0,
        exposure: None, // Using pre_exposure instead
        reactive_mask: None,
        transparency_and_composition: None,
        flags: FsrDispatchFlags::empty(),
    };

    let mut view = context.create_view(&render_queue, max_render_size, max_upscale_size);

    println!("render_size: {render_size:?}, upscale_size: {max_upscale_size:?}");
    println!("FSR3BB, max_render_size: {max_render_size:?}, max_upscale_size: {max_upscale_size:?}");

    // Execute FSR3
    context
        .dispatch(&mut view, encoder, &mut dispatch_info)
        .expect("FSR3 dispatch failed");
}

// impl ViewNode for Fsr3Node {
//     type ViewQuery = (
//         &'static ExtractedCamera,
//         &'static ExtractedView,
//         &'static ViewTarget,
//         &'static ViewPrepassTextures,
//         &'static Fsr3,
//         &'static Fsr3RenderContext,
//         &'static Fsr3Textures,
//         &'static TemporalJitter,
//         &'static MainPassResolutionOverride,
//         &'static Msaa,
//     );

//     fn run(
//         &self,
//         _graph: &mut RenderGraphContext,
//         render_context: &mut bevy_render::renderer::RenderContext,
//         (
//             camera,
//             view,
//             view_target,
//             prepass_textures,
//             fsr3,
//             fsr3_context,
//             fsr3_textures,
//             temporal_jitter,
//             resolution_override,
//             msaa,
//         ): QueryItem<Self::ViewQuery>,
//         world: &World,
//     ) -> Result<(), NodeRunError> {
//         if *msaa != Msaa::Off {
//             warn!("FSR3 requires MSAA to be disabled");
//             return Ok(());
//         }

//         let (Some(prepass_motion_vectors_texture), Some(prepass_depth_texture)) =
//             (&prepass_textures.motion_vectors, &prepass_textures.depth)
//         else {
//             return Ok(());
//         };

//         let render_queue = world.resource::<RenderQueue>();
//         let time = world.resource::<Time>();

//         let view_target = view_target.post_process_write();

//         let upscale_size = view.viewport.zw();
//         let render_size = resolution_override.0;

//         // Extract camera parameters from the projection matrix
//         // For perspective projection (infinite reverse z):
//         // clip_from_view[3][3] == 0.0 indicates perspective
//         // near plane is at clip_from_view[3][2]
//         // fov can be derived from clip_from_view[1][1] (which is f = 1/tan(fov/2))
//         let clip_from_view = view.clip_from_view;

//         let (camera_fov_y, camera_near, camera_far) = if clip_from_view.w_axis.w == 0.0 {
//             // Perspective projection
//             let f = clip_from_view.y_axis.y;
//             let fov_y = 2.0 * (1.0 / f).atan();
//             let near = clip_from_view.w_axis.z;
//             // For infinite far plane (reversed z), far is f32::INFINITY
//             let far = f32::INFINITY;
//             (fov_y, far, near)
//         } else {
//             warn!("FSR3 requires a perspective camera projection");
//             return Ok(());
//         };

//         // Calculate motion vector scale
//         // Bevy's motion vectors are in render resolution, FSR3 expects them in pixels
//         let motion_vector_scale = [-(render_size.x as f32), -(render_size.y as f32)];

//         let mut context = fsr3_context.context.lock().unwrap();

//         // Create a command encoder specifically for FSR3
//         let encoder = render_context.command_encoder();

//         // Build dispatch info - wgpu_ffx expects raw wgpu types
//         let mut dispatch_info = FsrDispatchInfo {
//             encoder,
//             queue: wgpu::Queue::clone(&render_queue),
//             color: wgpu::Texture::clone(&view_target.source_texture),
//             depth: wgpu::Texture::clone(&prepass_depth_texture.texture.texture),
//             motion_vectors: wgpu::Texture::clone(&prepass_motion_vectors_texture.texture.texture),
//             dilated_depth: wgpu::Texture::clone(&fsr3_textures.dilated_depth.texture),
//             dilated_motion_vectors: wgpu::Texture::clone(
//                 &fsr3_textures.dilated_motion_vectors.texture,
//             ),
//             reconstructed_previous_depth: wgpu::Buffer::clone(
//                 &fsr3_textures.reconstructed_previous_depth,
//             ),
//             output: wgpu::Texture::clone(&view_target.destination_texture),
//             render_size: [render_size.x, render_size.y],
//             upscale_size: [upscale_size.x, upscale_size.y],
//             jitter_offset: [temporal_jitter.offset.x, temporal_jitter.offset.y],
//             motion_vector_scale,
//             camera_fov_y,
//             camera_near,
//             camera_far,
//             frame_time_delta: time.delta_secs() * 1000.0,
//             reset_history: fsr3.reset,
//             enable_sharpening: fsr3.enable_sharpening,
//             sharpness: fsr3.sharpness,
//             pre_exposure: 1.0, // TODO: integrate with auto-exposure
//             view_space_to_meters_factor: 1.0,
//             exposure: None, // Using pre_exposure instead
//             reactive_mask: None,
//             transparency_and_composition: None,
//             flags: FsrDispatchFlags::empty(),
//         };

//         println!("FSR3BB");

//         // Execute FSR3
//         context
//             .dispatch(&mut dispatch_info)
//             .expect("FSR3 dispatch failed");

//         Ok(())
// }
// }
