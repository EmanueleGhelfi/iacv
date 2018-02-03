%% This script exports variables inside a txt file
format long
diary('out.txt');
diary on

disp("Horizontal Vanishing Line");

l_inf_prime

disp("Angle from left to right horizontal face");

theta

disp("Rotation matrix from left to right.");
disp("This is the rotation of the right");
disp("Reference frame in the left reference frame.");

R_from_left_to_right

disp("Position of the right reference frame in the left reference frame");

relative_pose_from_left_to_right

disp("Distance from left to right");

relative_position

disp("Camera Intrinsic Parameters");

K

fx

fy

alfa

u0

v0

disp("Rotation of the left face in the camera frame");

R

disp("Position of the left face in the camera frame");

T

disp("Rotation of the camera in the frame of the left horizontal face");

cameraRotation

disp("3D Position of the camera in the frame of the left horizontal face");

cameraPosition

disp("Rotation of the right face in the camera frame");

R_from_cam_to_right_plane

disp("Position of the right face in the camera frame");

T_from_cam_to_right_plane

disp("Rotation of the camera in the frame of the right horizontal face");

camera_orientation_wrt_right

disp("3D Position of the camera in the frame of the right horizontal face");

cameraPos_wrt_right

disp("3D coordinates of the left face");
left_face

disp("3D coordinates of the right face");
right_face

disp("3D coordinates of the points on the vertical faces. ");
disp("First four points are on the left vertical face.");
disp("The other four are on the right vertical face.");
disp("Coordinates are in the frame of the left horizontal face.");

vert_points

diary off
