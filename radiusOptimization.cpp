#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <iomanip>

using namespace std;
using namespace Eigen;

// Fit polar coordinates to a circle of best fit
tuple<double, double, double> fitCircleFlat(const vector<double>& radii, const vector<double>& angles) {
    int n = radii.size();
    MatrixXd A(n, 3);
    VectorXd b(n);

    for (int i = 0; i < n; ++i) {
        double r = radii[i];
        double theta = angles[i];
        A(i, 0) = 2 * r * cos(theta);
        A(i, 1) = 2 * r * sin(theta);
        A(i, 2) = 1;
        b(i) = r * r;
    }

    VectorXd x = A.fullPivHouseholderQr().solve(b);

    double x0 = x(0);
    double y0 = x(1);
    double radius = sqrt(x(2) + x0 * x0 + y0 * y0);

    return make_tuple(x0, y0, radius);
}

// Fit round, cartesian points to a circle of best fit
tuple<double, double, double, double> fitCircleRound(const vector<double>& x_values, const vector<double>& z_values, double y_value) {
    int n = x_values.size();
    MatrixXd A(n, 3);
    VectorXd b(n);

    for (int i = 0; i < n; ++i) {
        A(i, 0) = 2 * x_values[i];
        A(i, 1) = 2 * z_values[i];
        A(i, 2) = 1;
        b(i) = x_values[i] * x_values[i] + z_values[i] * z_values[i];
    }

    MatrixXd aT = A.transpose();
    VectorXd solution = (aT * A).colPivHouseholderQr().solve(aT * b);

    double x0 = solution(0);
    double z0 = solution(1);
    double radius = sqrt(abs(x0 * x0 + z0 * z0 - solution(2)));

    return make_tuple(x0, z0, radius, y_value);
}

// Calculate the line of best fit within the object
Vector3d lineOfBestFit(const vector<Vector3d>& points, const Vector3d& center, const string& filename) {
    Matrix3d covariance = Matrix3d::Zero();
    ofstream outfile(filename);

    for (const auto& point : points) {
        Vector3d deviation = point - center;
        covariance += deviation * deviation.transpose();
    }
    covariance /= points.size();

    // Perform SVD and extract the direction vector of the line from the first column of the U matrix
    JacobiSVD<MatrixXd> svd(covariance, ComputeThinU | ComputeThinV);
    Vector3d direction_vector = svd.matrixU().col(0);

    cout << "Direction vector of the line:\n" << direction_vector << endl;
    direction_vector.normalize();
    for (double y = -360.0; y <= 360.0; y += 1.2) { // Adjust spacing as needed
        double fy = (y - center.y()) / direction_vector.y();
        double x = center.x() + direction_vector.x() * fy;
        double z = center.z() + direction_vector.x() * fy;
		outfile << std::setprecision(10) << x << " " << y << " " << z << endl;
    }

	outfile.close();
    return direction_vector;
}

Vector3d projectPointOntoPlane(const Vector3d& point, const Vector3d& planePoint, const Vector3d& planeNormal) {
	Vector3d v = point - planePoint;
	double distance = v.dot(planeNormal);
	return point - distance * planeNormal;
}

/// <summary>
/// expands a cylinder to its maximum radius within the pointcloud
/// </summary>
/// <param name="pointCloud">Log Data</param>
/// <param name="planePoint">Point to anchor the planar projection (Center of smallest circle)</param>
/// <param name="planeNormal">Normal vector to the plane (Direction vector of linear interpolation)</param>
/// <returns></returns>
Vector4d ExpandCylinder(const std::vector<Vector3d>& pointCloud,
	const Vector3d& planePoint,
	const Vector3d& planeNormal) {

	Vector3d Uguess(0, 0, 1);
	Vector3d U1 = Uguess - Uguess.dot(planeNormal) * planeNormal;
	U1.normalize();
	Vector3d V1 = U1.cross(planeNormal);
	V1.normalize();
	MatrixXd UV = MatrixXd(2, 3);
	UV << U1.transpose(), V1.transpose();

	ofstream outfile("2dprojectpoints.txt");
	std::vector<Vector2d> projectedPoints;
	for (const auto& point : pointCloud) {
		Vector3d projectedPoint = projectPointOntoPlane(point, planePoint, planeNormal);
		projectedPoints.push_back(UV * projectedPoint);
		outfile << projectedPoints.back()(0) << " " << projectedPoints.back()(1) << " " << "0" << endl;
	}
	outfile.close();

	//Expand circle to first point
	double radius = 10000;
	Vector2d CircleCenter = UV * planePoint;

	Vector2d closestPoint1;
	for (const auto& point : projectedPoints) {
		double distance = (point - CircleCenter).norm();
		if (distance < radius) {
			radius = distance;
			closestPoint1 = point;
		}
	}

	ofstream outfile2("2dCircle and center initial.txt");
	outfile2 << std::setprecision(10) << CircleCenter(0) << " " << CircleCenter(1) << " " << 0 << endl;
	for (double theta = 0; theta <= 2 * M_PI; theta += 2 * M_PI / 1000) { // Adjust spacing as needed
		double x = CircleCenter(0) + radius * sin(theta);
		double y = CircleCenter(1) + radius * cos(theta);
		double z = 0;
		outfile2 << std::setprecision(10) << x << " " << y << " " << z << endl;
	}
	outfile2.close();

	//expand circle to second point
	Vector2d du;
	du(0) = CircleCenter(0) - closestPoint1(0);
	du(1) = CircleCenter(1) - closestPoint1(1);
	du.normalize();
	double closestdistance1 = 10000;
	Vector2d closestPoint2;
	for (const auto& point : projectedPoints) {
		double distance = -(pow((closestPoint1(0) - point(0)), 2) + pow((closestPoint1(1) - point(1)), 2)) / (2 * du(0) * (closestPoint1(0) - point(0)) + 2 * du(1) * (closestPoint1(1) - point(1)));
		if (distance < closestdistance1 && distance>0) {
			closestdistance1 = distance;
			closestPoint2 = point;
		}
	}
	radius = closestdistance1;
	CircleCenter = du.array() * radius + closestPoint1.array();

	ofstream outfile3("2dCircle and center halfway.txt");
	outfile3 << std::setprecision(10) << CircleCenter(0) << " " << CircleCenter(1) << " " << 0 << endl;
	for (double theta = 0; theta <= 2 * M_PI; theta += 2 * M_PI / 1000) { // Adjust spacing as needed
		double x = CircleCenter(0) + radius * sin(theta);
		double y = CircleCenter(1) + radius * cos(theta);
		double z = 0;
		outfile3 << std::setprecision(10) << x << " " << y << " " << z << endl;
	}
	outfile3.close();

	//expand to third point
	Vector2d v = (closestPoint2 - closestPoint1).normalized();
	Vector2d BasePoint = (closestPoint1 + closestPoint2) / 2;
	du = (CircleCenter - BasePoint).normalized();
	//Vector2d BasePoint = CircleCenter - radius * du;

	double currentDistance = (CircleCenter - BasePoint).norm();

	double closestdistance2 = 10000;
	Vector2d closestPoint3;
	for (const auto& point : projectedPoints) {
		//calculate perpendicular bisector
		Vector2d midpoint = (closestPoint1 + point) / 2.0;
		Vector2d direction_vector = point - closestPoint1;

		// Calculate direction vector of the perpendicular bisector
		Vector2d bisector_vector = Vector2d(-direction_vector.y(), direction_vector.x());
		// The perpendicular bisector passes through the midpoint
		Vector2d bisector_point = midpoint;

		double denominator = du.x() * bisector_vector.y() - du.y() * bisector_vector.x();
		Vector2d diff = bisector_point - BasePoint;
		double distance = (diff.x() * bisector_vector.y() - diff.y() * bisector_vector.x()) / denominator;

		if (distance < closestdistance2 && distance>currentDistance && point != closestPoint1 && point != closestPoint2) {
			closestdistance2 = distance;
			closestPoint3 = point;
		}
	}

	CircleCenter = du.array() * closestdistance2 + BasePoint.array();

	radius = (closestPoint3 - CircleCenter).norm();

	ofstream outfile4("2dCircle and center final.txt");
	outfile4 << std::setprecision(10) << CircleCenter(0) << " " << CircleCenter(1) << " " << 0 << endl;
	for (double theta = 0; theta <= 2 * M_PI; theta += 2 * M_PI / 1000) { // Adjust spacing as needed
		double x = CircleCenter(0) + radius * sin(theta);
		double y = CircleCenter(1) + radius * cos(theta);
		double z = 0;
		outfile4 << std::setprecision(10) << x << " " << y << " " << z << endl;
	}
	outfile4.close();

	//turn center back to 3d point and return it
	Vector3d newCenter = Vector3d::Zero(3);//UV.inverse() * CircleCenter;
	newCenter = U1 * CircleCenter(0) + V1 * CircleCenter(1);
	Vector4d results;
	results << newCenter, radius;
	return results;
}

void PrintCircle(const Vector3d& CenterPoint, const double& radius, ofstream& filestream) {
    for (double theta = 0; theta <= 2*M_PI; theta += 2 * M_PI/1000) { // Adjust spacing as needed
        double y = CenterPoint.y();
        double x = CenterPoint.x() + radius * sin(theta);
        double z = CenterPoint.z() + radius * cos(theta);
        filestream << std::setprecision(10) << x << " " << y << " " << z << endl;
    }
}

// formats a matrix of radii from block files and converts the data into cartesian coordinates
vector<Vector3d> reformatBlocks(ifstream& file) {
	int rows, cols;
	file >> rows >> cols;
	file.ignore();

	double angleInc = 360.0 / rows;
	double lengthInc = 101.0 / 32; // divide between 32 lasers
	vector<Vector3d> xyzData;
	double angle = 0.0;

	for (int i = 0; i < rows; ++i) {
		string line;
		getline(file, line);
		istringstream iss(line);
		double length = 0.0;
		
		for (int j = 0; j < cols; ++j) {
			double radius = 0.0;
			iss >> radius;
			if (radius != 0.000000) {
				xyzData.push_back(Vector3d(radius * sin(angle * 4 * acos(0.0) / 360), length, radius * cos(angle * 4 * acos(0.0) / 360)));
			}
			length += lengthInc;
		}
		angle += angleInc;
	}

	return xyzData;
}

int main() {
	string filename = "outputCenteredLogRound.txt";
	ifstream file(filename);
	map<double, vector<double>> column_groups_first;
	map<double, vector<double>> column_groups_third;
	vector<Vector3d> data_points;

	// if the input is a block file, reformat the data
	if (filename.find("Blk") != string::npos || filename.find("blk") != string::npos) {
		vector<Vector3d> blockPoints = reformatBlocks(file);
		ofstream outfile("blockPoints.txt");

		// write data to file for CloudCompare visualization
		for (int i = 0; i < blockPoints.size(); ++i) {
			outfile << blockPoints[i][0] << " " << blockPoints[i][1] << " " << blockPoints[i][2] << endl;
		}

		outfile.close();

		// read into column_groups_first and column_groups_third for circle fitting
		for (const auto& point : blockPoints) {
			column_groups_first[point[1]].push_back(point[0]);
			column_groups_third[point[1]].push_back(point[2]);
			data_points.push_back(point);
		}
	}
	else {
		for (string line; getline(file, line);) {
			istringstream iss(line);
			double first_value, second_value, third_value;

			// group data by rings on the y-axis
			if (iss >> first_value >> second_value >> third_value) {
				column_groups_first[second_value].push_back(first_value);
				column_groups_third[second_value].push_back(third_value);
				data_points.push_back(Vector3d(first_value, second_value, third_value));
			}
		}
	}

	vector<tuple<double, double, double, double>> circleParams;

    for (const auto& pair : column_groups_first) {
        double y_value = pair.first;
        vector<double> x_values = pair.second;
        vector<double> z_values = column_groups_third[y_value];

		circleParams.push_back(fitCircleRound(x_values, z_values, y_value)); // choose either fitCircleFlat or fitCircleRound depending on input data
		cout << "Circle Center (x0, y0, z0): " << get<0>(circleParams.back()) << ", " << get<3>(circleParams.back()) << ", " << get<1>(circleParams.back()) << endl;
		cout << "Circle Radius: " << get<2>(circleParams.back()) << endl;

		// visualize the circles of best fit for CloudCompare
		// ofstream radiusOutputFile("radiusOfEachCircle.txt");
		// PrintCircle(Vector3d(get<0>(circleParams.back()), get<3>(circleParams.back()), get<1>(circleParams.back())), get<2>(circleParams.back()), outputFile);
		// radiusOutputFile.close();
    }

	Vector3d center = Vector3d::Zero();
	double smallest_radius = 0.0;

	for (const auto& circle : circleParams) {
		if (get<2>(circle) < smallest_radius || smallest_radius == 0) {
			center << get<0>(circle), get<3>(circle), get<1>(circle);
            smallest_radius = get<2>(circle);
        }
	}

    cout << "Smallest radius: " << smallest_radius << endl;
	cout << "Center of smallest circle:\n" << center << endl;

    Vector3d LBF = lineOfBestFit(data_points, center, "lineOfBestFitRoundCentroid.txt");
	Vector4d projectedpoints = ExpandCylinder(data_points, center, LBF);
	center = projectedpoints.head<3>();
	double radius = projectedpoints(3);
	ofstream outfile("cylinderPoints.txt");

	for (int i = -500; i < 500; ++i) {
		Vector3d point = center + i * (LBF); // Equally spaced points along the line
		for (double theta = 0; theta <= 2 * M_PI; theta += 2 * M_PI / 1000) { // Adjust spacing as needed
			Vector3d initialPoint;
			initialPoint.y() = 0/*point.y()*/;
			initialPoint.x() = /*point.x()*/ radius * sin(theta);
			initialPoint.z() = /*point.z()*/ radius * cos(theta);

			double yaw = atan(LBF.x() / LBF.y());
			double pitch = atan(LBF.z() / sqrt((LBF.x() * LBF.x()) + (LBF.y() * LBF.y())));
			double roll = 0;
			Matrix3d Rx, Ry, Rz;
			Rx << 1, 0, 0, 0, cos(pitch), -sin(pitch), 0, sin(pitch), cos(pitch);
			Ry << cos(yaw), 0, sin(yaw), 0, 1, 0, -sin(yaw), 0, cos(yaw);
			Rz << cos(roll), -sin(roll), 0, sin(roll), cos(roll), 0, 0, 0, 1;

			Vector3d rotated_point = Rz * Ry * Rx * initialPoint;
			rotated_point += point;
			outfile << std::setprecision(10) << rotated_point(0) << " " << rotated_point(1) << " " << rotated_point(2) << endl;
		}
	}
	outfile.close();

    return 0;
}