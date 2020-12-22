import bpy
import numpy as np
import mathutils
from datetime import datetime
import os
import colorsys
import OpenEXR
import array  # for transforming read exr to python-usable
# import imageio
import yaml
from datetime import datetime

print('##### Starting script #####')
####################################################################################
######################### Blend-file specific Configuration ########################
# TODO: Change path to this file
basepath = '/home/adashao/Documents/DL-Robotics-Depth-Estimation/datageneration'

# Make sure blender knows and uses available GPU devices
bpy.context.preferences.addons['cycles'].preferences.get_devices()

# Main camera used for rendering
cam = bpy.data.objects['CameraOrbbec']
# List of objects that are spawned and animated
object_list = bpy.data.collections['AnimatedPrototypes'].objects.keys()

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Set this true if you want to save original object materials as yaml file
dump_materials = False
if dump_materials:
    orig_materials = {}
    # Record the original materials of every object in the scene to reset segmentation if needed
    for obj in bpy.data.objects:
        if hasattr(obj.data, 'materials') and len(obj.data.materials) > 0:
            # print(obj.name)
            orig_materials[obj.name] = obj.data.materials[0].name

    orig_materials['datetime'] = datetime_str
    with open(basepath + '/material_cfg.yaml', 'w') as outfile:
        yaml.dump(orig_materials, outfile, default_flow_style=False)

####################################################################################
######################### Configuration ############################################
####################################################################################

 # You can update this config file to change render parameters
with open(basepath + '/ALR_Praktikum_cfg.yaml', 'r') as cfg_file:
    cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

######################### Output Configuration #####################################
# bpy.data.node_groups["Compositing Nodetree"].nodes['img_with_Z'].format.exr_codec = 'ZIP'


######################### Render Configuration #####################################
bpy.context.scene.render.engine = cfg['render']['engine']
bpy.context.scene.render.fps = cfg['render']['fps']
# bpy.context.scene.render.image_settings.color_depth = '16'
bpy.context.scene.render.use_file_extension = False

if cfg['render']['engine'] == "BLENDER_EEVEE":
    bpy.context.scene.eevee.taa_render_samples = cfg['render']['samples']
elif cfg['render']['engine'] == "CYCLES":

    bpy.context.scene.cycles.samples = cfg['render']['samples']
    bpy.context.scene.view_layers['View Layer'].cycles['use_denoising'] = cfg['render']['denoising']

    bpy.context.scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            cprefs.compute_device_type = compute_device_type
            print("FabUseComputeDevice:", compute_device_type)
            break
        except TypeError:
            print("FabTypeError for", compute_device_type)

    # Enable all CPU and GPU devices
    for device in cprefs.devices:
        device.use = True

# bpy.context.scene.render.filepath = cfg['render']['filepath']  # FabMaybeNotNecessary

bpy.data.scenes['Scene'].render.resolution_x = cfg['render']['resolution_x']
bpy.data.scenes['Scene'].render.resolution_y = cfg['render']['resolution_y']

######################### Animation space values ###################################
np.random.seed(datetime.now().microsecond)
# Original camera position and rotation (above the rubber plate)
or_loc = mathutils.Vector(cfg['space_val']['camera']['loc'].values())
or_rot = mathutils.Quaternion(cfg['space_val']['camera']['quat_rot'].values())
print(f"or_rot is {or_rot}")
# Space above rubber plate in which camera moves around
cam_rbp_space = cfg['space_val']['cam_mov_space']

# Space above rubber plate in which objects spawn
obj_rbp_space = cfg['space_val']['obj_spawn_space']

####################################################################################
"""
Reset methods
"""

def reset_cam():
    print(f"reset_camera_{cam.__dir__()}")
    cam.animation_data_clear()
    cam.location = or_loc
    cam.rotation_quaternion = or_rot

    for con in cam.constraints.items():
        cam.constraints.remove(con[1])


# Deletes all objects in the AnimatedObjects collection
def reset_anim_objects():
    # Deselect all currently selected objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)

    for obj in bpy.data.collections['AnimatedObjects'].all_objects:
        obj.select_set(True)
    bpy.ops.object.delete(use_global=False, confirm=False)


# Resets start and end frame to default
def reset_frames():
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 240


# Reset all markers on colour_ramp for all object classes
def reset_colour_ramps():
    for obj in object_list:
        # Get colour ramp
        mat_name = 'Seg_' + obj
        shader_nodes = bpy.data.materials[mat_name].node_tree.nodes
        ramp = shader_nodes["SegColour"].color_ramp

        # Clear any elements - cannot remove the last element
        for i in range(len(ramp.elements) - 1): ramp.elements.remove(ramp.elements[-1])
        # Set last element's position to 0
        ramp.elements[0].position = 0


# Reset shader materials which were possibly changed for segmentation purposes
def reset_materials():
    # First, set the background to use nodes (as opposed to grey colour) again:
    bpy.context.scene.world.use_nodes = True

    # Get the original material config file
    with open(basepath + '/material_cfg.yaml', 'r') as cfg_file:
        orig_materials = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    for obj in bpy.data.objects:

        # For objects in AnimatedObjects collection, the original material might not be saved;
        # just take the prototype's material
        if obj.name in bpy.data.collections['AnimatedObjects'].all_objects.keys():
            prototype = obj.name[:-4]  # Remove the automatic numbering
            obj.data.materials[0] = bpy.data.objects[prototype].data.materials[0]

        elif hasattr(obj.data, 'materials') and len(obj.data.materials) > 0:

            # Loop through all materials attached to object
            for m in range(len(obj.data.materials)):
                # Iterate through keys of dict with * and turn to indexable list
                mat_name = [*orig_materials[obj.name]][m]
                obj.data.materials[m] = bpy.data.materials.get(mat_name)


# Resets the links in the compositing tree
def reset_comp_tree_links():
    bpy.data.scenes[0].node_tree.links.clear()


# Resets the power of the scene lights and strength of bg image
def reset_lighting():
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = \
        cfg['experiment']['background']['strength']

    for light in bpy.data.collections['Lights'].all_objects.values():
        if 'Area' in light.name:
            light.data.energy = cfg['experiment']['area_lights']['power']
            light.data.color = (cfg['experiment']['area_lights']['color'],) * 3
        else:
            light.data.energy = cfg['experiment']['sun_light']['power']
            light.data.color = (cfg['experiment']['area_lights']['color'],) * 3


####################################################################################
def randomize_lighting():
    """
    Randomizes the energy of the lights in the scene and the background strength.
    Randomization is done through a uniform distribution centered around mean given by the config.
    """
    bg_mean = cfg['experiment']['background']['strength']
    bg_dev = cfg['experiment']['background']['max_dev']

    bg_str = np.random.uniform(low=bg_mean - bg_dev, high=bg_mean + bg_dev)

    color_mean = cfg['experiment']['area_lights']['color']
    color_dev = cfg['experiment']['area_lights']['color_dev']

    for light in bpy.data.collections['Lights'].all_objects.values():
        if 'Area' in light.name:
            light_mean = cfg['experiment']['area_lights']['power']
            light_dev = cfg['experiment']['area_lights']['power_dev']
        else:
            light_mean = cfg['experiment']['sun_light']['power']
            light_dev = cfg['experiment']['sun_light']['power_dev']

        light.data.energy = np.random.uniform(low=light_mean - light_dev, high=light_mean + light_dev)
        light.data.color = tuple(np.random.uniform(low=color_mean - color_dev, high=color_mean + color_dev, size=3))


def randomize_rbplate_texture():
    """
    Chooses a random texture for the 'RubberPlate' object out of the 'Plate_bg_' materials
    Chooses 2 random colors to further modify the material; every Plate_bg material contains
    2 color nodes named 'ran_rgb1' and 'ran_rgb2' that modify its color
    """

    mats = bpy.data.materials.keys()
    material = bpy.data.materials[np.random.choice([m for m in mats if 'Plate_bg' in m])]

    col_nodes = [material.node_tree.nodes['rgb_ran1'],
                 material.node_tree.nodes['rgb_ran2']]

    for n in col_nodes:
        ran_rgb = colorsys.hsv_to_rgb(np.random.uniform(0, 1),  # Cover all hue range
                                      np.random.uniform(0.5, 1),  # Only choose saturation values bigger than 0.5
                                      1)

        n.outputs[0].default_value = ran_rgb + (1,)

    bpy.data.objects['RubberPlate'].data.materials[0] = material


def change_bg_image(img_name=None, path=None):
    """
    Changes background image to random image provided in cfg or image specified by
    img_name
    If path is set, first loads the image specified by path
    """
    if path is not None:
        img = bpy.data.images.load(path, check_existing=True)
    else:
        img = bpy.data.images[img_name]

    bpy.data.worlds['World'].node_tree.nodes['Environment Texture'].image = img


def set_colour_ramps():
    """
    Set the segmentation colour ramp properties for every object class
    Each object class has its segmentation material and its own hue, different
    instances receive different saturation levels
    """
    num_class = len(object_list)
    max_num_instances = cfg['experiment']['num_spawned_obj_instances']  # Maximum number of object instances per class

    # Get num_class * max_num_instances RGB colours, ordered by class
    colours = get_rgb_colours(num_class, max_num_instances)

    reset_colour_ramps()

    for obj, class_idx in zip(object_list, range(num_class)):
        # Get colour ramp
        mat_name = 'Seg_' + obj
        shader_nodes = bpy.data.materials[mat_name].node_tree.nodes
        ramp = shader_nodes["SegColour"].color_ramp

        # print(mat_name)
        # Set the dividing value to max number of spawned objects
        shader_nodes["max_inst_idx"].outputs[0].default_value = max_num_instances
        # Move the object position in colour ramp to centre of colour intervals to avoid border cases
        shader_nodes["add_half_pos"].inputs[1].default_value = 1 / (2 * max_num_instances)

        for i in range(max_num_instances):
            # Add new color ramp part at position pos
            pos = i / max_num_instances
            ramp.elements.new(pos)

            # We always use same alpha=1 value
            alpha = 1
            ramp.elements[-1].color = colours[class_idx][i] + (alpha,)


def get_rgb_colours(num_hues, num_levels):
    """
    Returns a set of num_hues * num_levels RGB values
    Due to polar-like coordinates in HSV, choose hues in HSV according to num_hues and
    saturation according to num_levels. Then translate to RGB with default brightness 1
    """

    colours = []
    # Make a list of hues between 0 and 1
    hues = [i / num_hues for i in range(num_hues)]

    # Make a list of saturation values from 0.5 to 1 - we don't want super unsaturated colours
    s_list = [1 - i / (2 * num_levels) for i in range(num_levels)]

    for h in hues:
        colours.append([])
        for s in s_list:
            # Always use brightness value 1
            colours[-1].append(colorsys.hsv_to_rgb(h, s, 1))

    return colours


def assign_segment_mat():
    """
    Assigns the "Segmentation" material to all objects in the AnimatedObjects collection and the
    "Segmantion_bg" to all other objects
    Also sets the world background appropriately
    This is used to produce a segmentation map of the camera view
    """
    bg_mat = bpy.data.materials.get('Segmentation_bg')
    for obj in bpy.data.objects:
        # TODO: Assign special material to robot
        # if not obj.parent == None and obj.parent.name == 'Panda':

        if hasattr(obj.data, 'materials') and len(obj.data.materials) > 0:
            for m in range(len(obj.data.materials)):
                obj.data.materials[m] = bg_mat

    for obj in bpy.data.collections['AnimatedObjects'].all_objects:
        prototype = obj.name[:-4]  # Remove the automatic numbering
        obj.data.materials[0] = bpy.data.materials.get('Seg_' + prototype)

    bpy.context.scene.world.use_nodes = False
    bpy.context.scene.world.color = bg_mat.node_tree.nodes['bg_colour'].color


def cam_track_object(camera, target):
    """
    Makes camera track object by adding a track constraint and choosing the axis
    such that camera faces the object upright
    """
    camera.constraints.new(type='TRACK_TO')
    camera.constraints["Track To"].target = bpy.data.objects[target]
    camera.constraints["Track To"].track_axis = 'TRACK_NEGATIVE_Z'
    camera.constraints["Track To"].up_axis = 'UP_Y'


def ran_loc(space_dict):
    """
    Returns a uniformly random location in a cube specified by space_dict
    space_dict should be a dictionary containting the entries
    xlow, xhigh, ylow, yhigh and zlow, zhigh
    """
    ran_loc = np.random.uniform([space_dict['xlow'], space_dict['ylow'], space_dict['zlow']],
                                [space_dict['xhigh'], space_dict['yhigh'], space_dict['zhigh']])
    # print(f"random location of the {ran_loc}")
    return mathutils.Vector(ran_loc)


def ran_quat_orientation():
    """
    Generates a random quaternion by generating 3 samples in (0,1) and constructing a
    quaternion from it
    """
    ran = np.random.uniform(size=3)
    u, v, w = ran[0], ran[1], ran[2]
    quat = (np.sqrt(1 - u) * np.sin(2 * np.pi * v),
            np.sqrt(1 - u) * np.cos(2 * np.pi * v),
            np.sqrt(u) * np.sin(2 * np.pi * w),
            np.sqrt(u) * np.cos(2 * np.pi * w))
    return mathutils.Quaternion(quat)


def spawn_animated_object(objectname='LetterB', num_obj_inst=3, random_obj=False, space_dict=obj_rbp_space):
    """
    Copies an object and spawns it uniform-randomly in the cube given by space_dict
    Links the object to the 'AnimatedObjects' collection
    space_dict should be a dictionary containting the entries
    xlow, xhigh, ylow, yhigh and zlow, zhigh
    """
    num_obj = len(object_list) * num_obj_inst
    for n in range(num_obj):
        if random_obj:
            objectname = np.random.choice(object_list)
        else:
            # If the objects are not randomized, just add the same number of objects for every class
            index = int(n / num_obj_inst)
            objectname = object_list[index]

        object = bpy.data.objects[objectname]

        # determine the location of the objects
        loc = ran_loc(space_dict)
        # determine the orientation of the objects
        rot_quat = ran_quat_orientation()

        new_obj = object.copy()
        new_obj.data = object.data.copy()
        new_obj.location = loc
        new_obj.rotation_mode = 'QUATERNION'
        new_obj.rotation_quaternion = rot_quat
        new_obj.pass_index = int(n % num_obj_inst)

        # print('### Creating object', objectname, new_obj.pass_index)
        bpy.data.collections['AnimatedObjects'].objects.link(new_obj)

        # (Re)link the object to make sure its physics work
        if new_obj in bpy.context.scene.rigidbody_world.collection.objects.items():
            bpy.context.scene.rigidbody_world.collection.objects.unlink(new_obj)
        bpy.context.scene.rigidbody_world.collection.objects.link(new_obj)


def animate_cam_predefined(start_frame=20, n_frames=5):
    """
    Makes predefined movements for the camera above the RubberPlate object and sets the keyframes accordingly.
    """
    # TODO original
    print(f"animate_cam_predefined.")
    # locs = [
    #     (-0.8, 0.0, 1.29),
    #     (-0.77, -0.02, 1.27),
    #     (-0.77, 0.02, 1.27),
    #     (-0.74, 0, 1.33),
    # ]
    locs = [
        (-0.8, 0.00, 1.29),
        (-0.8, 0.001, 1.29),
        (-0.8, 0.002, 1.29),
        (-0.8, 0.003, 1.29),
    ]

    current_frame = start_frame
    for loc in locs:
        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)
        cam.location = loc
        print(f"location is {cam.location}")
        # print('Inserting keyframe at ', current_frame, loc)
        # Insert location and rotation keyframe
        # bpy.ops.transform.translate(value=loc - cam.location)
        # print(f"attrs of camera are {cam.__dir__()}")
        print(f"rotation of camera are {cam.rotation_quaternion}")
        cam.keyframe_insert(data_path="location", frame=current_frame)
        current_frame += 1

        if current_frame > start_frame + n_frames:
            break


def ran_animate_cam(num_moves=3, start_frame=10, last_frame=250):
    """
    Makes randomized movements for the camera above the RubberPlate object and sets the keyframes accordingly.
    The time for each key frame is determined by Gaussians centered
    evenly spaced between start_frame and the last frame.
    The movement goals are evenly placed on a donut around the center of the rubber plate with a uniform radius.
    """
    int("ran_animate_cam is running.")
    current_frame = start_frame
    cam.keyframe_insert(data_path="location", frame=current_frame)
    cam.keyframe_insert(data_path="rotation_euler", frame=current_frame)

    # The average time interval between keyframes/movement targets
    avg_time_interval = (last_frame - start_frame) / num_moves
    min_time_interval = avg_time_interval / 2

    # Calculate evenly distributed locations on ring around center of RubberPlate
    avg_angle_diff = 2 * np.pi / (num_moves + 1)
    min_angle_diff = avg_angle_diff / 2
    current_angle = 0  # np.pi #Start position

    # Randomized factor to either go clockwise or counter clockwise
    # angle_fac = np.random.choice([-1, 1])
    ## TODO original
    angle_fac = 0

    center_x = bpy.data.objects['RubberPlate'].location[0]
    center_y = bpy.data.objects['RubberPlate'].location[1]
    # Intervals in which radius and z are uniformly sampled from
    radius_interval = cfg['space_val']['cam_mov_space']['radius']
    z_interval = cfg['space_val']['cam_mov_space']['z']

    # Insert keyframe for every move
    for mov in range(num_moves + 1):

        # current_angle += angle_fac * sample_normal_with_minimum(
        #     mean=avg_angle_diff,
        #     std=cfg['space_val']['cam_mov_space']['angle_dev'],
        #     min_int=min_angle_diff)
        print(f"current angle is {current_angle}")
        radius = np.random.uniform(radius_interval[0], radius_interval[1])
        z = np.random.uniform(z_interval[0], z_interval[1])
        print(f"z axis is {z}")
        x = radius * np.cos(current_angle) + center_x
        y = radius * np.sin(current_angle) + center_y

        loc = mathutils.Vector((x, y, z))

        bpy.ops.object.select_all(action='DESELECT')
        cam.select_set(True)

        # print('Inserting keyframe at ', current_frame, loc)
        # Insert location and rotation keyframe
        bpy.ops.transform.translate(value=loc - cam.location)
        cam.keyframe_insert(data_path="location", frame=current_frame)

        # Sample the time frame for the next location goal; if it is the last move,
        # just choose the last frame as time goal to idle time at end
        if mov == num_moves - 1:
            current_frame = last_frame
        else:
            current_frame += sample_normal_with_minimum(mean=avg_time_interval,
                                                        std=cfg['experiment']['cam_time_frame_std'],
                                                        min_int=min_time_interval)


def sample_normal_with_minimum(mean, std, min_int):
    """
    Sample interval by sampling a Normal distribution parametrized by mean and std
    with minimum interval min_int
    """
    interval = np.NINF

    while interval < min_int:
        interval = np.random.normal(loc=mean, scale=std)

    return interval


def stop_obj_animation():
    """
    Stops the animation of the AnimatedObjects collection and set the object's rotation
    and location to the last (animated) rotation & location
    """
    for obj in bpy.data.collections['AnimatedObjects'].all_objects:
        # obj.keyframe_insert(data_path="location", frame=99)
        # obj.keyframe_insert(data_path="rotation_euler", frame=99)
        obj.rigid_body.enabled = False

        # Set object pose to last animated pose
        # obj.location = obj.matrix_world.translation
        # obj.rotation_euler = obj.matrix_world.to_euler()

        # tmp = obj.rotation_euler.to_matrix().to_4x4()
        # obj.rotation_euler = (tmp @ obj.matrix_world).to_euler()
        # Disable physics animation by setting kinematic (GUI: "Animated) to True


def get_cam_intrinsics():
    """
    Returns the camera instrinsics containing fx, fy, cy, cy in a format compatible with
    opencv (as 3x3 matrix)
    Taken from: https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/
    """
    scene = bpy.context.scene
    assert scene.render.resolution_percentage == 100
    #  assume angles describe the horizontal field of view
    assert cam.data.sensor_fit != 'VERTICAL'

    f_in_mm = cam.data.lens
    sensor_width_in_mm = cam.data.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - cam.data.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * cam.data.shift_y

    K = [[f_x, 0, c_x],
         [0, f_y, c_y],
         [0, 0, 1]]

    return K


def render(start_frame, n_frames, save_pose_depth=True, path=None):
    """
    Small helper function to actually render
    If save_pose_depth, saves poses and depth information
    """

    w = bpy.data.scenes['Scene'].render.resolution_x
    h = bpy.data.scenes['Scene'].render.resolution_y

    for f in range(start_frame, start_frame + n_frames):
        bpy.context.scene.frame_set(f)

        bpy.ops.render.render(write_still=False)

        if save_pose_depth and cfg['experiment']['mode']['depth']:
            pass
            # Get depth information
            # print("FabDebug: Get depth information skipped")
            # depth_exr = OpenEXR.InputFile('{}/depth/depth{:04d}.exr'.format(path, f))
            # depth_img = np.reshape(array.array('f', depth_exr.channel('Z.V')), (h, w))
            #
            # # Remove unreasonable background values:
            # depth_img[depth_img == 10000000000.0] = np.NaN
            #
            # depth_undistorted = undistort_depth(depth_img)
            #
            # exr_out = OpenEXR.OutputFile('{}/depth/depth{:04d}.exr'.format(path, f),
            #                              depth_exr.header())
            # exr_out.writePixels({'Z.V': depth_undistorted.astype('float32').tostring()})

        if save_pose_depth and cfg['experiment']['mode']['pose']:
            # Save camera world coordinates and intrinsics
            pose_output = {'cam': np.array(cam.matrix_world).tolist()}

            for obj in bpy.data.collections['AnimatedObjects'].all_objects:
                # Translate object world coordinates to cam coordinates
                obj_pose, bb = to_cam_coord(obj, invert_z=True, bb=True)

                # Save pose and pass_index as object instance identifier
                pose_output[obj.name] = {'matrix': obj_pose.tolist(),
                                         'bb': bb.tolist(),
                                         'id': obj.pass_index}

            with open('{}/poses/pose{:04d}.yaml'.format(path, f), 'w') as outfile:
                yaml.dump(pose_output, outfile, default_flow_style=False)

        print("### Finished rendering frame {} ###".format(f))


def render_and_save(start_frame=0, n_frames=240, seq_idx=None):
    """
    First renders the normal image with engine specified in cfg
    If depth, renders the depth image with EEVEE engine to avoid distortion due
    to cycles pinhole camera
    If segmentation, colours the animated objects and renders again TODO
    """
    # print("FabDebug: render_and_save, seq_idx:", seq_idx)

    cfg['render_start'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg['render']['cam_intr'] = get_cam_intrinsics()
    cfg['experiment']['objects'] = object_list

    # Reset all links in compositioning tree to avoid unwanted output
    reset_comp_tree_links()

    # Get all output nodes
    node_tree = bpy.data.scenes[0].node_tree
    render_node = node_tree.nodes["Render Layers"]

    if cfg['experiment']['mode']['sequence']:
        output_dir = os.path.join(cfg['output']['path'], 'Seq{:03d}'.format(seq_idx))
        os.makedirs(output_dir)

        # save preliminary config:
        with open(output_dir + '/cfg.yaml', 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)
    else:
        # Use fixed output path for all images
        output_dir = cfg['output']['path']
        os.makedirs(output_dir, exist_ok=True)

        path = os.path.join(cfg['output']['path'], 'images', 'im{:04d}gt.png')
        i = 0
        while os.path.exists(path.format(i)):
            i += 1
        output_filename_rgb = 'im{:04d}gt.png'.format(i)
        output_filename_depth = 'im{:04d}depthgt.exr'.format(i)

    print("FabDebug: output_dir created:", output_dir)

    # Run the physics simulation until start:
    for i in range(start_frame):
        bpy.context.scene.frame_set(i)
    # Next, stop the physics animation so that objects don't move during rendering
    stop_obj_animation()

    """
    For every render option, the following has to be done:
    1. Choose the appropriate output node 
    2. Set the output path of said node
    3. Link the appropriate render output to said node
    4. Render
    """

    # Make output directories for poses and depth
    if cfg['experiment']['mode']['pose']:
        os.makedirs(output_dir + '/poses/', exist_ok=True)
    if cfg['experiment']['mode']['depth']:
        os.makedirs(output_dir + '/depth/', exist_ok=True)

    #
    if cfg['experiment']['mode']['image'] or cfg['experiment']['mode']['depth']:
        png_node = node_tree.nodes['img_png']
        png_node.base_path = output_dir + "/images/"
        # png_node.name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not cfg['experiment']['mode']['sequence']:
            png_node.file_slots[0].path = output_filename_rgb  # specify image output filename
        png_node.format.color_depth = cfg['render']['color_depth']  # number of bits per color channel
        node_tree.links.new(render_node.outputs['Image'], png_node.inputs[0])

    if cfg['experiment']['mode']['depth']:
        # Set depth output to viewer node for later postprocessing
        exr_node = node_tree.nodes['depth_exr']
        if cfg['experiment']['mode']['sequence']:
            exr_node.base_path = output_dir + "/depth/imdepthgt.exr"  # im{:04d}depthgt.exr
        else:
            exr_node.base_path = output_dir + "/depth/" + output_filename_depth
        node_tree.links.new(render_node.outputs['Depth'], exr_node.inputs[0])

    if cfg['experiment']['mode']['image'] or cfg['experiment']['mode']['depth'] or cfg['experiment']['mode']['pose']:
        render(start_frame, n_frames, path=output_dir)
        print("Finished rendering of normal image")

    reset_comp_tree_links()

    if cfg['experiment']['mode']['segment']:
        """
        Next, render segmentation image with special materials for each object
        """

        # Set output path for native output and custom output nodes:
        png_node = node_tree.nodes['img_png']
        png_node.base_path = output_dir + "/segment/"
        png_node.format.color_depth = '8'  # Segmentation doesn't have a lot of colours
        assign_segment_mat()
        node_tree.links.new(render_node.outputs['Image'],
                            png_node.inputs[1])

        render(start_frame, n_frames, save_pose_depth=False)
        print("Finished rendering depth image (and segmentation)")

    # Dump render/experiment config
    cfg['render_end'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    with open(output_dir + '/cfg.yaml', 'w') as outfile:
        yaml.dump(cfg, outfile, default_flow_style=False)

    # rename output files
    if cfg['experiment']['mode']['image']:
        image_dir = os.path.join(output_dir, 'images')
        # print("Rename image_dir:", image_dir)
        files = sorted(os.listdir(image_dir))
        matches = [file for file in files if not file.endswith(".png")]

        image_path = os.path.join(image_dir, 'im{:04d}gt.png')
        i = 0
        while os.path.exists(image_path.format(i)):
            i += 1
        for match in matches:
            src = os.path.join(image_dir, match)
            dst = os.path.join(image_dir, 'im{:04d}gt.png'.format(i))
            os.rename(src, dst)
            i += 1
    if cfg['experiment']['mode']['depth']:
        depth_dir = os.path.join(output_dir, 'depth')
        files = sorted(os.listdir(depth_dir))
        matches = [file for file in files if not file.endswith(".exr")]

        depth_path = os.path.join(depth_dir, 'im{:04d}depthgt.exr')
        i = 0
        while os.path.exists(depth_path.format(i)):
            i += 1
        for match in matches:
            src = os.path.join(depth_dir, match)
            dst = os.path.join(depth_dir, 'im{:04d}depthgt.exr'.format(i))
            os.rename(src, dst)
            i += 1


def undistort_depth(depth_img):
    """
    Transforms a radially distorted depth image base on a pinhole camera to orthographic depth
    It simply calculates the pixel distance of every pixel to center, uses that to calculate the
    angle of the corresponding ray and scales the depth accordingly
    """
    cam_intr = cfg['render']['cam_intr']
    f_pix = cam_intr[0][0]
    cx = cam_intr[0][2]
    cy = cam_intr[1][2]

    pixel_coord = np.mgrid[:depth_img.shape[0], :depth_img.shape[1]]

    # Distance of every pixel to the center in pixel units
    d_pix = np.sqrt((pixel_coord[1, :, :] - cx) ** 2 + (pixel_coord[0, :, :] - cy) ** 2)
    # arctan gives the ray angle here
    depth_undistorted = np.multiply(np.cos(np.arctan(d_pix / f_pix)), depth_img)

    return depth_undistorted


def to_cam_coord(object, invert_z=False, bb=False):
    """
    Translates object's coordinates to camera coordinate system and returns as np array
    If invert_z, changes the sign of the returned z position; this should be used when
    exporting as objects captured by the camera have a negative z value in the camera's
    coordinate system
    If bb, also returns object's bounding box in camera coordinates
    """
    cam_coord_inv = np.array(cam.matrix_world.inverted())
    object_coord = np.array(object.matrix_world)

    ret = cam_coord_inv @ object_coord

    if invert_z:
        ret[2, :] *= -1

    # Normalize rotation part
    for axis in range(3):
        dir = ret[:, axis]
        dir /= np.linalg.norm(dir)

    # Get bounding box coordinates
    if bb:
        bb = np.array(object.bound_box).T
        # Add row of ones so we can multiply it with world matrices
        bb = np.vstack((bb, np.ones(8)))
        # Bounding box is in object-local coordinate system; translate it to global
        bb = object_coord @ bb
        # Lastly, translate the bounding box to camera coord
        bb = cam_coord_inv @ bb

        bb = np.delete(bb, 3, axis=0)  # Remove the rows of ones

        if invert_z:
            bb[2, :] *= -1

    return ret, bb


def run(reset=True, animate=False, render=False, segmentation=False, randomize=False, seq_idx=None):
    """
    Runs different components of the programs in an aggregated way. Each component indicated
    by the arguments runs several associated functions
    """
    if reset:
        reset_cam()
        reset_anim_objects()
        reset_materials()
        reset_frames()
        # cam_track_object(cam, 'RubberPlate')
        # print("Finished resetting")

    if randomize:
        randomize_rbplate_texture()
        randomize_lighting()

    if animate:
        spawn_animated_object(random_obj=False,
                              num_obj_inst=cfg['experiment']['num_spawned_obj_instances'],
                              space_dict=obj_rbp_space)
        cam_track_object(cam, "RubberPlate")
        if cfg['experiment']['camera_movement_predefined']:
            print("follow predefined camera trajectory")
            animate_cam_predefined(cfg['experiment']['start_frame'],
                                   cfg['experiment']['n_frames'])
        else:
            print("follow random camera trajectory")
            ran_animate_cam(cfg['experiment']['num_camera_moves'],
                            cfg['experiment']['start_frame'],
                            cfg['experiment']['n_frames'])

        # print("Finished animating")

    if render:
        # print("Starting render")
        render_and_save(start_frame=cfg['experiment']['start_frame'],
                        n_frames=cfg['experiment']['n_frames'],
                        seq_idx=seq_idx)
        # print("Finished rendering & saving all")

    if segmentation:
        assign_segment_mat()


def main():
    """
    Main method to create & render multiple scenes
    Configuration is entirely basepath & config name at the start and config; the rest is
    saved in the blend file and need not be changed unless there are fundamental setup changes
    Every background image is used twice: Once the original and once the same
    image rotated around z-axis by 90°
    """
    cfg['output']['path'] += '_' + datetime_str
    if not cfg['experiment']['mode']['sequence']:
        if cfg['experiment']['n_frames'] > 1:
            print("FabWarning: Sequence flag was not set even though a sequence length is given!")
            cfg['experiment']['n_frames'] = 1

    img_path, _, bg_imgs = next(os.walk(cfg['input']['bg_imgs']))

    # Filter out any non .hdr images:
    bg_imgs = [i for i in bg_imgs if '.hdr' in i]

    bg_imgs.sort()

    import random
    random.shuffle(bg_imgs)

    # Main loop for creating different scenes/videos
    for n in range(cfg['experiment']['scene_start_idx'],
                   cfg['experiment']['scene_start_idx'] + cfg['experiment']['n_scenes']):

        if n >= 4 * len(bg_imgs):
            print('### !!! Ending dataset rendering, no more background images !!! ###')
            break

        bg_img = bg_imgs[int(n / 4)]
        bg_img_path = os.path.join(img_path, bg_img)

        # print("bg_img", bg_img)
        # print("bg_img_path", bg_img_path)
        # print("bg_imgs", bg_imgs)

        """
        Randomizes background image, rubber plate material and lighting for every scene
        The latter two are done in run()
        """
        change_bg_image(path=bg_img_path)
        cfg['experiment']['bg_img_used'] = bg_img
        # print(f"using {bg_img} to generate new background.")
        # For 4 scenes, background is either original, flipped, rotated or flipped and rotated
        if n % 4 == 0:
            cfg['experiment']['bg_img_used'] = bg_img
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Rotation'].default_value[2] = 0
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Scale'].default_value[1] = 1
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].rotation[2] = 0
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].scale[1] = 1
        elif n % 3 == 0:
            cfg['experiment']['bg_img_used'] = bg_img + '_rot_flop'
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Rotation'].default_value[2] = np.pi/2
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Scale'].default_value[1] = -1
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].rotation[2] = np.pi/2
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].scale[1] = -1
        elif n % 2 == 0:
            cfg['experiment']['bg_img_used'] = bg_img + '_rot'
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Rotation'].default_value[2] = np.pi/2
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Scale'].default_value[1] = 1
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].rotation[2] = np.pi/2
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].scale[1] = 1
        else:
            cfg['experiment']['bg_img_used'] = bg_img + '_flop'
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Rotation'].default_value[2] = 0
            bpy.data.worlds['World'].node_tree.nodes["Mapping"].inputs['Scale'].default_value[1] = -1
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].rotation[2] = 0
            # bpy.data.worlds['World'].node_tree.nodes["Mapping"].scale[1] = -1

        #        run(reset=True, animate=True, render=True, segmentation=True, randomize=True)
        run(reset=True, animate=True, render=True, segmentation=False, randomize=True, seq_idx=n)

    if not cfg['experiment']['mode']['sequence']:
        # Dump render/experiment config
        cfg['render_end'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with open(cfg['output']['path'] + '/cfg.yaml', 'w') as outfile:
            yaml.dump(cfg, outfile, default_flow_style=False)


# reset_colour_ramps()
# set_colour_ramps()
# assign_segment_mat()
# reset_comp_tree _links()
# reset_materials()
# change_bg_image(path='/home/bol7tue/git/pcl_pose_estimation/dat/hdri_background/xiequ_yuan_4k_flop.hdr')
# stop_obj_animation()
# cam_track_object(cam, 'RubberPlate')
# reset_lighting()
# randomize_lighting()
# randomize_rbplate_texture()
# reset_cam()
# cam_track_object(cam, 'RubberPlate')
# ran_animate_cam(cfg['experiment']['num_camera_moves'], cfg['experiment']['start_frame'], cfg['experiment']['n_frames'])

main()

# run()

# run(reset=True, animate=True, render=False, segmentation=False)
# run(reset=True, animate=True, render=False, segmentation=True)
# run(reset=True, animate=True, render=True, segmentation=False)
# run(reset=True, animate=True, render=True, segmentation=True)
# run(reset=False, animate=False, render=True, segmentation=True)


print('##### Finished script animate_camera.py #####')