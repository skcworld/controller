"""Launch file for controller manager node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    """Setup launch based on mode and env arguments."""
    
    mode = LaunchConfiguration('mode').perform(context)
    env = LaunchConfiguration('env').perform(context)
    
    # Get package share directory
    controller_share = FindPackageShare('controller').find('controller')
    
    # Determine controller type (MAP, PP, or AUG)
    controller_type = mode.upper()
    
    # Determine environment (sim or real)
    is_sim = env.lower() == 'sim'
    
    # Dynamic topic remappings based on environment
    # In sim: topics are published as /ego_racecar/odom, etc.
    # In real: topics are published as /odom, etc.
    if is_sim:
        odom_remapping = ('/odom', '/ego_racecar/odom')
        frenet_odom_remapping = ('/frenet/odom', '/car_state/frenet/odom')
        map_frame_param = 'map'
        base_link_frame_param = 'ego_racecar/base_link'
    else:
        odom_remapping = ('/odom', '/odom')
        frenet_odom_remapping = ('/frenet/odom', '/frenet/odom')
        map_frame_param = 'map'
        base_link_frame_param = 'base_link'
    
    # Determine which parameter file to use
    if controller_type == 'MAP':
        if is_sim:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'l1_params_sim.yaml'
            ]).perform(context)
        else:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'l1_params.yaml'
            ]).perform(context)
    elif controller_type == 'PP':
        if is_sim:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'pp_params_sim.yaml'
            ]).perform(context)
        else:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'pp_params.yaml'
            ]).perform(context)
    elif controller_type == 'AUG':
        if is_sim:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'aug_params_sim.yaml'
            ]).perform(context)
        else:
            params_file = PathJoinSubstitution([
                controller_share,
                'config',
                'aug_params.yaml'
            ]).perform(context)
    else:
        raise ValueError(f"Invalid mode: {controller_type}. Must be 'MAP', 'PP', or 'AUG'")
    
    # Determine lookup table based on environment (only for MAP controller)
    # PP and AUG controllers do not use lookup table, they use geometric formula
    if controller_type == 'MAP':
        if is_sim:
            lookup_table_file = PathJoinSubstitution([
                controller_share,
                'config',
                'SIM_linear_lookup_table.csv'
            ]).perform(context)
        else:
            lookup_table_file = PathJoinSubstitution([
                controller_share,
                'config',
                'RBC1_pacejka_lookup_table.csv'
            ]).perform(context)
    else:
        # PP and AUG controllers don't need lookup table, but we provide a dummy path
        lookup_table_file = PathJoinSubstitution([
            controller_share,
            'config',
            'SIM_linear_lookup_table.csv'
        ]).perform(context)
    
    
    # Controller manager node
    controller_node = Node(
        package='controller',
        executable='controller_manager',
        name='controller_manager',
        output='screen',
        parameters=[
            {
                'mode': controller_type,
                'lookup_table_path': lookup_table_file,
                'l1_params_path': params_file if controller_type == 'MAP' else '',
                'pp_params_path': params_file if controller_type == 'PP' else '',
                'aug_params_path': params_file if controller_type == 'AUG' else '',
                'map_frame': map_frame_param,
                'base_link_frame': base_link_frame_param,
                'use_sim_time': False,  # Use wall clock time instead of simulation time
            }
        ],
        remappings=[
            ('/local_waypoints', '/local_waypoints'),
            odom_remapping,
            frenet_odom_remapping,
            ('/sensors/imu/raw', '/sensors/imu/raw'),
            ('/drive', '/drive'),
        ]
    )
    
    return [controller_node]


def generate_launch_description():
    """Generate launch description for controller manager."""
    
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='MAP',
        description='Controller mode: MAP, PP, or AUG'
    )
    
    env_arg = DeclareLaunchArgument(
        'env',
        default_value='real',
        description='Environment: real or sim'
    )
    
    return LaunchDescription([
        mode_arg,
        env_arg,
        OpaqueFunction(function=launch_setup)
    ])
