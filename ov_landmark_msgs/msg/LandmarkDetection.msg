# One detected landmark object in an image
string type                   # object type in the landmark map (e.g. gate_blue), "aruco" for ArUco markers
int32 id                      # known identity (ArUco marker id), -1 = unknown: matched to the map in the filter
LandmarkKeypoint[] keypoints  # in the order the type defines (ascending landmark id, ArUco: corners 0..3)
float32 box_area              # pixels^2, widens the matching gate of near objects
