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
// Change original x and z axes to x and y within this function for simplicity
tuple<double, double, double> fitCircleRound(const vector<double>& x_values, const vector<double>& y_values) {
    int n = x_values.size();
    MatrixXd A(n, 3);
    VectorXd b(n);

    for (int i = 0; i < n; ++i) {
        A(i, 0) = -2 * x_values[i];
        A(i, 1) = -2 * y_values[i];
        A(i, 2) = 1;
        b(i) = x_values[i] * x_values[i] + y_values[i] * y_values[i];
    }

    MatrixXd aT = A.transpose();
    VectorXd solution = (aT * A).colPivHouseholderQr().solve(aT * b);

    double x0 = solution(0);
    double y0 = solution(1);
    double radius = sqrt(abs(x0 * x0 + y0 * y0 - solution(2)));

    return make_tuple(x0, y0, radius);
}

// Calculate the line of best fit within the object
vector<Vector3d> lineOfBestFit(const vector<Vector3d>& points, const string& filename) {
    Vector3d centroid = Vector3d::Zero();
    Matrix3d covariance = Matrix3d::Zero();
    ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open file for writing." << endl;
    }

    for (const auto& point : points) {
        centroid += point;
    }
    centroid /= points.size();

    for (const auto& point : points) {
        Vector3d deviation = point - centroid;
        covariance += deviation * deviation.transpose();
    }
    covariance /= points.size();

    // Perform SVD and extract the direction vector of the line from the first column of the U matrix
    JacobiSVD<MatrixXd> svd(covariance, ComputeThinU | ComputeThinV);
    Vector3d direction_vector = svd.matrixU().col(0);

    cout << "Direction vector of the line:\n" << direction_vector << endl;
    cout << "Centroid of the point cloud:\n" << centroid << endl;

    for (double y = -360.0; y <= 360.0; y += 1.2) { // Adjust spacing as needed
        double x = centroid.x() + direction_vector.x() * y;
        double z = centroid.z() + direction_vector.z() * y;
        outputFile << std::setprecision(10) << x << " " << y << " " << z << endl;
    }

    outputFile.close();
    cout << "Points written to file: " << filename << endl;
    vector<Vector3d> result = { direction_vector , centroid };

    return result;
}

int main() {
	ifstream file("outputCenteredLogRound.txt");
    map<double, vector<double>> column_groups_first;
    map<double, vector<double>> column_groups_third;
    vector<Vector3d> data_points;
    vector<double> radius;
    double smallest_radius = 0;

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

	vector<Vector3d> LBF = lineOfBestFit(data_points, "lineOfBestFitRound.txt");

    for (const auto& pair : column_groups_first) {
        double key = pair.first;
        vector<double> x_values = pair.second;
        vector<double> z_values = column_groups_third[key];

		auto circleParams = fitCircleRound(x_values, z_values); // choose either fitCircleFlat or fitCircleRound depending on input data
        cout << "Circle Center (x0, y0): " << get<0>(circleParams) << ", " << get<1>(circleParams) << endl;
        cout << "Circle Radius (r): " << get<2>(circleParams) << endl;
        radius.push_back(get<2>(circleParams));
    }

    smallest_radius = *min_element(radius.begin(), radius.end());
    cout << "Smallest radius: " << smallest_radius << endl;

    return 0;
}