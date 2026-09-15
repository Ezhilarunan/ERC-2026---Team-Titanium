from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Capture the arguments from the terminal command
    shelf_arg = DeclareLaunchArgument(
        'shelf_column_number', 
        default_value='3',
        description='Target column number (1-5)'
    )
    
    color_arg = DeclareLaunchArgument(
        'book_colour', 
        default_value='red',
        description='Target book color (red, blue, green, yellow)'
    )

    # 2. Pass them to your node as parameters
    solution_node = Node(
        package='team_titanium',       # Your package name
        executable='main_node',       # The name of your compiled python script
        name='team_titanium_node',
        parameters=[{
            'shelf_column_number': LaunchConfiguration('shelf_column_number'),
            'book_colour': LaunchConfiguration('book_colour')
        }]
    )

    return LaunchDescription([
        shelf_arg,
        color_arg,
        solution_node
    ])