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
        A(i, 0) = -2 * x_values[i];
        A(i, 1) = -2 * z_values[i];
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
Vector3d lineOfBestFit(const vector<Vector3d>& points, const string& filename, const Vector3d& center) {
    //Vector3d centroid = Vector3d::Zero();
    Matrix3d covariance = Matrix3d::Zero();
    ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open file for writing." << endl;
    }

    /*for (const auto& point : points) {
        centroid += point;
    }
    centroid /= points.size();*/

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
        outputFile << std::setprecision(10) << x << " " << y << " " << z << endl;
    }

    outputFile.close();
    cout << "Points written to file: " << filename << endl;

    return direction_vector;
}

void PrintCircle(const Vector3d& CenterPoint, const double& radius, ofstream filestream) {
    for (double theta = 0; theta <= 2*M_PI; theta += 2 * M_PI/1000) { // Adjust spacing as needed
        double y = CenterPoint.y();
        double x = CenterPoint.x() + radius * sin(theta);
        double z = CenterPoint.z() + radius * cos(theta);
        filestream << std::setprecision(10) << x << " " << y << " " << z << endl;
    }
}

int main() {
	ifstream file("outputCenteredLogRound.txt");
    map<double, vector<double>> column_groups_first;
    map<double, vector<double>> column_groups_third;
	vector<tuple<double, double, double, double>> circleParams;
    vector<Vector3d> data_points;
    Vector3d center = Vector3d::Zero();
    double smallest_radius = 0.0;

    for (string line; getline(file, line);) {
        istringstream iss(line);
        double first_value, second_value, third_value;
        Vector3d temp_vector;
        // group data by rings on the y-axis
        if (iss >> first_value >> second_value >> third_value) {
            column_groups_first[second_value].push_back(first_value);
            column_groups_third[second_value].push_back(third_value);
			temp_vector << first_value, second_value, third_value;
        }
		data_points.push_back(temp_vector);
    }

    for (const auto& pair : column_groups_first) {
        double y_value = pair.first;
        vector<double> x_values = pair.second;
        vector<double> z_values = column_groups_third[y_value];

		circleParams.push_back(fitCircleRound(x_values, z_values, y_value)); // choose either fitCircleFlat or fitCircleRound depending on input data
		cout << "Circle Center (x0, y0, z0): " << get<0>(circleParams.back()) << ", " << get<3>(circleParams.back()) << ", " << get<1>(circleParams.back()) << endl;
		cout << "Circle Radius: " << get<2>(circleParams.back()) << endl;
    }

	// find the smallest radius within circleParams and return all circle parameters
	for (const auto& circle : circleParams) {
		if (get<2>(circle) < smallest_radius || smallest_radius == 0) {
			center << get<0>(circle), get<3>(circle), get<1>(circle);
            smallest_radius = get<2>(circle);
        }
	}

	cout << "Smallest radius: " << smallest_radius << endl;
	cout << "Center of smallest circle:\n" << center << endl;

    Vector3d LBF = lineOfBestFit(data_points, "lineOfBestFitRoundCentroid.txt", center);

    ofstream outputFile("centerPoint.txt");
	outputFile << center.x() << " " << center.y() << " " << center.z() << endl;

    return 0;
}