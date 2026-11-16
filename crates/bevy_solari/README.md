# Bevy Solari

[![License](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](https://github.com/bevyengine/bevy#license)
[![Crates.io](https://img.shields.io/crates/v/bevy_solari.svg)](https://crates.io/crates/bevy_solari)
[![Downloads](https://img.shields.io/crates/d/bevy_solari.svg)](https://crates.io/crates/bevy_solari)
[![Docs](https://docs.rs/bevy_solari/badge.svg)](https://docs.rs/bevy_solari/latest/bevy_solari/)
[![Discord](https://img.shields.io/discord/691052431525675048.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2)](https://discord.gg/bevy)

![Logo](../../assets/branding/bevy_solari.svg)

## Runtime shadow terminator setting

The experimental correction is part of `bevy_solari`, not an example or a special
scene. Use it with an existing scene that already uses `SolariPlugins` and
`SolariLighting` cameras. No example modifications or built-in key bindings are
required.

`SolariLightingPlugin` (included in `SolariPlugins`) automatically initializes and
registers the reflected `SolariLightingSettings` resource. Existing resources are
preserved, so it can be configured before or after adding the plugins:

```rust,ignore
use bevy::{prelude::*, solari::prelude::*};

App::new()
    .insert_resource(SolariLightingSettings {
        shadow_terminator_fix: Some(true),
    })
    .add_plugins((DefaultPlugins, SolariPlugins));
// Keep the application's existing scene, camera, and update systems.
```

For runtime A/B comparison, change this resource from your existing UI or inspector,
with `ResMut<SolariLightingSettings>` in a system, or through the app world:

```rust,ignore
// Enable globally on the next extracted frame.
app.world_mut()
    .resource_mut::<SolariLightingSettings>()
    .shadow_terminator_fix = Some(true);

// Disable globally, including cameras with their local switch enabled.
app.world_mut()
    .resource_mut::<SolariLightingSettings>()
    .shadow_terminator_fix = Some(false);

// Release the global override and use each camera's own setting (the default).
app.world_mut()
    .resource_mut::<SolariLightingSettings>()
    .shadow_terminator_fix = None;
```

For per-camera control instead, leave the global override at `None` and change the
reflected `SolariLighting::shadow_terminator_fix` boolean on the existing camera:

```rust,ignore
fn enable_for_cameras(mut cameras: Query<&mut SolariLighting>) {
    for mut lighting in &mut cameras {
        lighting.shadow_terminator_fix = true;
    }
}
```

The global override does not overwrite camera components. Changing an effective
camera setting automatically resets its Solari history without reallocating the
world cache or recompiling shaders. Default behavior is unchanged: the global
override is `None`, and each camera's correction is `false`.

### Prototype scope

This is a deliberately limited, unoptimized validation of Johannes Hanika's
[Ray Tracing Gems II, chapter 4](https://jo.dreggn.org/home/2021_terminator.pdf):

- An extra camera ray recovers the first visible triangle and barycentric
  coordinates. Vertex tangent-plane projections determine a shadow-origin offset.
- Only **primary directional-light NEE shadow rays** use the offset. A scene lit
  exclusively by emissive area lights will not show a corrected terminator.
  Original shading positions, BRDFs, area-light shadows, indirect rays, and
  world-cache sampling are unchanged. This avoids mixing modified area-light
  visibility with an unmodified BRDF-emissive MIS strategy.
- Perspective cameras only. Missing/mismatched ray hits or invalid normals fall
  back to the original origin. Position matching uses a heuristic tolerance, not
  a guaranteed raster/ray geometry identity check.
- When enabled, ReSTIR spatial visibility uses the same correction, but temporal
  reuse is disabled: the previous TLAS alone cannot safely recover previous mesh
  data after instance slots or geometry have changed. Spatial reuse can add more
  recovery rays. Compare with ReSTIR off before judging performance or noise.
- Does not fix normal-map bump terminators, low-poly silhouettes, or all indirect
  terminators. Watch for light leaks, detached contact shadows, and hard-edge
  discontinuities. The reference pathtracer is unchanged. This is not a
  physically exact or production-ready solution.
