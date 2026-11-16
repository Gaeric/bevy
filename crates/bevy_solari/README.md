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

### Implementation: dedicated offset textures (scheme 3)

```text
existing raster depth/G-buffer (unchanged)
    -> offset generation compute pass
       one recovery ray per visible pixel + Gems II vertex-plane projection
    -> current offset texture: xyz = world-space displacement, w = match validity
    -> primary directional-light visibility + ReSTIR spatial/temporal visibility
    -> copy current offsets to the previous-offset texture after lighting
```

The producer and consumer are separate. The current producer is a dedicated Solari
compute pass, **not** a modification to Bevy's general raster prepass. It still uses
an additional primary ray to recover triangle information; scheme 3 does not by
itself eliminate that cost. Unlike the previous inline experiment, recovery is
performed only once per visible pixel and reused by all lighting/reuse queries.
A future raster or visibility-buffer producer can populate the same displacement
texture without changing those consumers.

ReSTIR temporal reuse is now supported while the correction is enabled: previous
pixel offsets are sampled at the same reprojected/permuted coordinate as previous
depth, G-buffer, and reservoirs. Reverse temporal visibility uses the stored offset
with the previous TLAS, without indexing current mesh data using an old instance ID.
Offsets are overwritten for every current pixel, including zero/invalid values for
background and mismatches. First frames, camera resets, toggles, resolution changes,
and skipped lighting frames invalidate history until a matching frame is available.

Dynamic switching remains a runtime uniform/resource choice, not a shader variant:

- **Enabled:** allocate current and previous `RGBA32Float` textures at the camera's
  internal rendering resolution. Generate offsets before lighting and copy history
  after all lighting reads. This costs 32 bytes per pixel for the pair (about
  **63.3 MiB at 1920x1080**, **253.1 MiB at 3840x2160**, per enabled camera), plus
  per-frame generation, sampling, and copy bandwidth.
- **Disabled:** skip generation and history copies and use zero displacement.
  Full-size textures are released and only two 1x1 binding placeholders remain.
- **Toggle:** replace only the offset textures and invalidate lighting history;
  existing world-cache, light-tile, and reservoir buffers are not reallocated.
- GPU diagnostics expose `solari_lighting/shadow_terminator_offsets` separately
  from lighting, so the producer's cost can be measured in the existing scene.

### Prototype scope

This is a limited implementation of Johannes Hanika's
[Ray Tracing Gems II, chapter 4](https://jo.dreggn.org/home/2021_terminator.pdf):

- Only **primary directional-light NEE shadow rays** use the offset. A scene lit
  exclusively by emissive area lights will not show a corrected terminator.
  Original shading positions, BRDFs, area-light shadows, indirect rays, and
  world-cache sampling are unchanged. This avoids mixing modified area-light
  visibility with an unmodified BRDF-emissive MIS strategy.
- Recovery starts at each pixel's reconstructed near-plane position, using the
  same inverse projection as depth reconstruction, including for orthographic
  projections. This does not change Solari's existing BRDF/view-direction logic.
- Missing/mismatched ray hits or invalid normals fall back to zero offset.
  Position matching uses a heuristic tolerance, not a guaranteed raster/ray
  geometry identity check. Nearly coincident surfaces, alpha-clipped geometry,
  and raster/ray deformation differences need particular care.
- Does not fix normal-map bump terminators, low-poly silhouettes, or all indirect
  terminators. Watch for light leaks, detached contact shadows, and hard-edge
  discontinuities. The reference pathtracer is unchanged. This is not a
  physically exact or production-ready solution.
