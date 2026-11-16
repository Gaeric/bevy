use super::{prepare::SolariLightingResources, SolariLighting, SolariLightingSettings};
use bevy_camera::Camera;
use bevy_ecs::system::{Commands, ResMut};
use bevy_pbr::deferred::SkipDeferredLighting;
use bevy_render::{sync_world::RenderEntity, MainWorld};

pub fn extract_solari_lighting(mut main_world: ResMut<MainWorld>, mut commands: Commands) {
    let settings = main_world
        .get_resource::<SolariLightingSettings>()
        .copied()
        .unwrap_or_default();
    let mut cameras_3d = main_world.query::<(RenderEntity, &Camera, Option<&mut SolariLighting>)>();

    for (entity, camera, solari_lighting) in cameras_3d.iter_mut(&mut main_world) {
        let mut entity_commands = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if let Some(mut solari_lighting) = solari_lighting
            && camera.is_active
        {
            entity_commands.insert((settings.apply(&solari_lighting), SkipDeferredLighting));
            solari_lighting.reset = false;
        } else {
            entity_commands.remove::<(
                SolariLighting,
                SolariLightingResources,
                SkipDeferredLighting,
            )>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::{system::RunSystemOnce, world::World};

    #[test]
    fn runtime_override_is_extracted_without_changing_camera_settings() {
        let mut render_world = World::new();
        let render_camera = render_world.spawn_empty().id();
        let mut main_world = MainWorld::default();
        main_world.insert_resource(SolariLightingSettings::default());
        let camera = main_world
            .spawn((
                RenderEntity::from(render_camera),
                Camera::default(),
                SolariLighting::default(),
            ))
            .id();
        render_world.insert_resource(main_world);

        render_world
            .run_system_once(extract_solari_lighting)
            .unwrap();
        assert!(
            !render_world
                .get::<SolariLighting>(render_camera)
                .unwrap()
                .shadow_terminator_fix
        );

        for enabled in [true, false] {
            render_world
                .resource_mut::<MainWorld>()
                .resource_mut::<SolariLightingSettings>()
                .shadow_terminator_fix = Some(enabled);
            render_world
                .run_system_once(extract_solari_lighting)
                .unwrap();
            assert_eq!(
                render_world
                    .get::<SolariLighting>(render_camera)
                    .unwrap()
                    .shadow_terminator_fix,
                enabled
            );
            assert!(
                !render_world
                    .resource::<MainWorld>()
                    .get::<SolariLighting>(camera)
                    .unwrap()
                    .shadow_terminator_fix
            );
        }

        // An explicit global false overrides a camera true; None restores camera control.
        render_world
            .resource_mut::<MainWorld>()
            .get_mut::<SolariLighting>(camera)
            .unwrap()
            .shadow_terminator_fix = true;
        render_world
            .run_system_once(extract_solari_lighting)
            .unwrap();
        assert!(
            !render_world
                .get::<SolariLighting>(render_camera)
                .unwrap()
                .shadow_terminator_fix
        );

        render_world
            .resource_mut::<MainWorld>()
            .resource_mut::<SolariLightingSettings>()
            .shadow_terminator_fix = None;
        render_world
            .run_system_once(extract_solari_lighting)
            .unwrap();
        assert!(
            render_world
                .get::<SolariLighting>(render_camera)
                .unwrap()
                .shadow_terminator_fix
        );

        // A global override also applies to cameras created after it was enabled.
        render_world
            .resource_mut::<MainWorld>()
            .resource_mut::<SolariLightingSettings>()
            .shadow_terminator_fix = Some(true);
        let new_render_camera = render_world.spawn_empty().id();
        render_world.resource_mut::<MainWorld>().spawn((
            RenderEntity::from(new_render_camera),
            Camera::default(),
            SolariLighting::default(),
        ));
        render_world
            .run_system_once(extract_solari_lighting)
            .unwrap();
        assert!(
            render_world
                .get::<SolariLighting>(new_render_camera)
                .unwrap()
                .shadow_terminator_fix
        );
    }
}
