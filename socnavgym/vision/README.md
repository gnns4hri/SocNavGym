# SocNavGym vision

Sensors on the robot (RGB-D cameras, lidars), placed with a URDF; the robot's joints in the action; a movable 3D view.
The 2D simulation is untouched: this package only looks at it and draws it in 3D (Blender, GPL v3, as SocNavGym).
An environment whose configuration has no `vision:` section (or one without sensors and viewer) is exactly as before and
never imports this package.

## Set-up (once)

The renderer is Blender's python module, which needs its own python (3.13 for Blender 5.3). It runs as a separate process,
started when the first environment with vision is reset.

    uv venv --python 3.13 .venv_render                                   # in the repository's root
    uv pip install --python .venv_render/bin/python bpy==5.3.0 numpy OpenEXR
    python3 socnavgym/vision/tools/fetch_assets.py                       # downloads the sources (220 MB) into vision_cache/
    .venv_render/bin/python socnavgym/vision/tools/build_assets.py       # makes socnavgym/vision/assets/ (37 MB)

Another location for the renderer's python: `vision.renderer.python` in the configuration, or `$SOCNAV_VISION_PYTHON`.

**Without the renderer** (no `bpy` in the renderer's python, or the 3D models not built) the simulator works in 2D as usual: an
environment whose configuration has a `vision:` section is still made, with a warning that says what is left out - the cameras,
the lidars in `mesh` mode and the 3D view. If nothing of the section is left, the environment is exactly the one without the
section (same observation, same action). A lidar in `simple` mode needs no renderer and stays, with the robot's joints.
`renderer: {required: true}` makes the missing renderer an error instead (for a training that cannot go on without its camera).

## Try it

    python -m socnavgym.vision.demo                           # windows: the 2D view with its menu, 3D view, camera, depth, segmentation, scan
    python -m socnavgym.vision.demo --no-window --save out    # no screen: pictures of 100 steps in out/
    python -m socnavgym.vision.trajectory_video file.json     # a recorded trajectory, seen from above, as file.mp4

Keys and mouse are listed by `--help`. The example configuration is `environment_configs/vision_demo.yaml`.

## Windows

`env.render()` always shows the simulator's 2D window. With a `vision:` section it has a bar at the top with the menu
**Windows**: a click opens the list of the rendered windows - `3D view` and one per picture of a sensor (`camera rgb`,
`camera depth`, `camera segmentation`, `scan`) - and a click on a name shows or hides that window (the list stays open; a click
elsewhere closes it). A window's own close button hides it as well. Only what can be rendered is listed; with nothing to list
there is no bar. The 3D view is not rendered while its window is hidden.

- At the start the 3D view is open (`viewer: {show: false}`: hidden) and the sensors' windows are hidden (`display: true`: open).
  `env.vision.shown` (name -> shown?) can be changed from a program.
- A sensor's window has the simulated sensor's resolution and cannot be resized.
- One size for the 2D window and the 3D view: `rendering: {resolution_x, resolution_y}`. The 2D window has exactly that size
  and the map is drawn in its middle, as large as it fits with its own proportions. The 3D view's window can be resized with the mouse and the view is rendered at the window's
  size; its time grows with the number of pixels (see Speed). `env.vision.view(width, height)` gives a picture of any size.
  (Configurations from before 2026-11-17 may have `viewer: {width, height}`: they are refused when loaded;
  `python socnavgym/vision/tools/remove_viewer_size.py FOLDER` takes the two keys out of every YAML file under a folder.)

## Configuration

    vision:
        robot:
            urdf: round_pan_tilt_robot        # a name in socnavgym/vision/robots (round_pan_tilt_robot: round, radius 0.25 m; box_pan_tilt_robot:
                                        # rectangular, 0.65 x 0.40 m), or a path (absolute or relative to the config file).
                                        # Left out: the one of the simulator's body (`robot: robot_shape`): round for
                                        # "circle", box for "rectangle". The model's size is the file's: it does not follow
                                        # robot_radius / robot_length / robot_width, which are what the simulator collides with
            packages: []                # folders where package://... meshes are looked for
            joints: {head_tilt: 0.15}   # starting positions; the others start at 0
        sensors:
            camera: {type: rgbd,  link: camera_link}
            scan:   {type: lidar, link: lidar_link, mode: simple}
        ground_truth: true              # segmentation, instance ids and sensor pose with every reading (also per sensor)
        display: false                  # true: the sensors' windows are open when env.render() is first called (the menu switches them)
        viewer: {enabled: true}         # also: samples, fov, azimuth, elevation, fit, margin, distance, target, follow, show
                                        # (its size is `rendering: resolution_x, resolution_y`)
        renderer: {samples: 24}         # also: engine (cycles | eevee), device, denoise, draft, bounces, threads, exposure, python, log, required
        scene: {wall_height: 2.5}       # also: table_height, chair_height, laptop_height, human_height, human_height_spread,
                                        # humans (list of models), lighting (auto | indoor | outdoor), floor_color, wall_color,
                                        # ceiling_color, lamp_power (20), lamp_spacing, lamp_color, sky_strength, floor_margin,
                                        # shadows (0 to 1, default 0.2: how dark the sun's shadows are; 1 = as the sky casts
                                        # them, 0 = none. What is in the sun looks the same whatever the value)

Only a sensor's `type` and `link` are required. A sensor is attached to a URDF link and looks along the link's +X, with +Z up.
An unknown field is an error (a mistyped name does not pass silently).

| sensor | model (default first) | fields that override the model |
|---|---|---|
| `rgbd` | `realsense_d435`, `realsense_d455`, `ideal` | `width`, `height`, `hfov` (degrees), `depth_min`, `depth_max`, `depth_noise`, `baseline`, `subpixel`, `samples`, `exposure` |
| `lidar` | `generic_2d`, `generic_3d_16`, `ideal` | `beams`, `fov`, `rings`, `vfov`, `range_min`, `range_max`, `noise`, `mode` (`simple` / `mesh`), `no_return` (`inf` / `max` / `zero`), `ignore_robot` |

Lidar modes: `simple` casts the beams against the simulator's own 2D shapes (people and plants are circles, tables, chairs and
walls rectangles; no renderer needed, about 1 ms); `mesh` casts them against the 3D models, animated people included
(beams pass between legs and under tables, as with a real scanner), and is the only mode for several rings.

## What the agent gets and gives

    obs["sensors"]["camera"] = {"rgb": (H, W, 3) uint8, "depth": (H, W) float32 metres along the optical axis, 0 = no reading,
                                "segmentation": (H, W) uint8, "instance": (H, W) int32, "pose": (4, 4) float32}
    obs["sensors"]["scan"]   = {"ranges": (beams,) or (rings, beams) float32 metres, "segmentation", "instance", "pose"}
    obs["joints"]            = (number of movable joints,) float32 positions

`segmentation` holds the class (`socnavgym.vision.ipc.CLASSES`: background, floor, wall, table, chair, plant, laptop, human,
robot, ceiling), `instance` the simulator's id of the entity (-1: none), `pose` the sensor's frame in the world (X forward,
Y left, Z up). `env.vision.sensor_info[name]` has the constants (camera matrix `K`, the lidar's angles, ranges).

The action is the simulator's three numbers followed by one number in [-1, 1] per movable joint of the URDF: the joint's
position target between its limits. Joints move towards their targets at most at the URDF's velocity limit. An action with
only three numbers is accepted and leaves the joints' targets as they were.

`env.vision.view()` returns the 3D view's picture; `env.vision.images()` the last readings as pictures;
`env.vision.viewer` is the orbiting camera (`target`, `distance`, `azimuth`, `elevation`, `follow`, `auto`).

## The 3D view

It starts as a bird's eye view: from straight above, the world's +X to the right and +Y up as in the simulator's 2D window,
framed so that everything is in the picture - the map, the walls, the furniture, the robot and every person. The framing is
redone for every picture (`auto`): when the robot or a person leaves the map (scenes without walls), the view widens to keep
them in, and it narrows again when they come back. In such scenes the floor also grows under whoever nears its edge, and the
default lighting is the open sky instead of a ceiling with lamps (`scene: {lighting: auto}`).

In its window: left drag (or the arrows, in any window) turns the view, and the framing goes on from the new direction; right drag (or
shift + left drag) moves it and the wheel (or + / -) zooms - from then on the view is the user's and no longer reframed;
F follows the robot; R goes back to the starting view. Configuration: `azimuth`, `elevation` (degrees; default -90, 90),
`margin` (room around the scene, default 1.08), `fit: false` (no automatic framing); giving `distance`, `target` or
`follow: true` also fixes the view unless `fit: true` is given.

## A video of a recorded trajectory

    python -m socnavgym.vision.trajectory_video trajectory.json [video.mp4]

renders a trajectory stored as JSON - the format of the SocNav datasets, which SocNavGym's sn26 rewards also record
(`sequence`: per instant the `timestamp`, the `robot`, the `people` and the `objects`; `walls`: lines) - with the 3D view from
straight above and saves it as MP4. The simulator is not run: the scene is built from the file alone. The view is framed once
on everything the trajectory touches, so the picture does not zoom during the video; the video plays in real time (`--fps`
changes that). Options: `--width`, `--height` (1920 x 1080), `--samples`, `--elevation`, `--azimuth` (a tilted view instead of
the bird's eye), `--wall-thickness`, `--wall-height`, `--robot` (URDF). The video is H.264 when `ffmpeg` is installed,
otherwise OpenCV's MPEG-4, which some players and browsers do not show.
Limits: tables, chairs, plants and laptops have 3D models, other kinds of object are left out (and named); an object stays
where it first appears; the goal is not drawn; the robot is the URDF's model, whatever the file says about its shape.

## The robot's URDF

Links, joints (fixed, revolute, continuous, prismatic) and visuals (box, cylinder, sphere, mesh: stl, obj, dae, glb / gltf)
are read. The root link lies on the floor at the simulator's robot pose. The URDF does not change the simulation: the
robot still collides as the simulator's circle.

## Speed and engines

On an RTX A6000 shared with other jobs: one 640x480 RGB-D frame takes about 0.3 to 0.4 s with `cycles` (path tracing with
OptiX, 24 samples, denoised), the 3D view (12 samples) about 0.2 s at 960x540 and 0.9 s at 1920x1080: on a slower
card make its window smaller or hide it. `cycles` on the processor alone takes about
20 s per frame: without a suitable graphics card use `engine: eevee` (Blender's rasteriser; lighting less true). `eevee` is
experimental: on the development server it took 0.8 s per frame and its process stopped without a message in some runs.

**Draft mode, for debugging**: `renderer: draft: true` (or the environment variable `SOCNAV_VISION_DRAFT=1`, which needs no
change to the YAML file) renders every picture at half its width and height, with at most 4 samples and no denoiser, and
enlarges it to the size asked for: the windows and the sensors' arrays keep their sizes, the pictures are coarse and grainy
(depth and labels are enlarged by repeating pixels, so they stay real values). A number sets the size (`draft: 0.3`). With a
1080 x 1080 3D view and a 640x480 camera a step took 0.78 s as usual, 0.35 s with `draft: true`, 0.30 s with `draft: 0.3`.
The lidar is not rendered and does not change. Not for recording data.

## Assets and licences

- furniture and surfaces: Poly Haven (CC0): metal_office_desk, SchoolChair_01, potted_plant_02, classic_laptop,
  laminate_floor_02, white_plaster_02;
- people: Microsoft Rocketbox (MIT): Male_Adult_01, Female_Adult_01 with their neutral walking and standing animations.

People walk in step with the distance they cover (no sliding feet) and stand with an idle motion when still.

## Known limits

- One model per kind of object and two people models; the simulator's sizes are applied to them (a 3 x 1.5 m table is a
  stretched desk).
- A person at a laptop stands where the 2D simulation puts them, which overlaps the table's 3D model.
- People who join or leave during an episode are not added to or removed from the 3D scene.
- The 3D view cannot see through walls; from low angles the near wall hides the room. From straight above people are
  seen as heads and shoulders: turn the view to see them properly.
