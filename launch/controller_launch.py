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
    
    # Determine controller type (MAP or PP)
    controller_type = mode.upper()
    
    # Determine environment (sim or real)
    is_sim = env.lower() == 'sim'
    
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
    else:
        raise ValueError(f"Invalid mode: {controller_type}. Must be 'MAP' or 'PP'")
    
    # Determine lookup table based on environment (only for MAP controller)
    # PP controller does not use lookup table, it uses geometric formula
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
                'NUC4_pacejka_lookup_table.csv'
            ]).perform(context)
    else:
        # PP controller doesn't need lookup table, but we provide a dummy path
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
                'map_frame': 'map',
                'base_link_frame': 'base_link',
            }
        ],
        remappings=[
            ('/local_waypoints', '/local_waypoints'),
            ('/odom', '/odom'),
            ('/frenet/odom', '/frenet/odom'),
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
        description='Controller mode: MAP or PP'
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
