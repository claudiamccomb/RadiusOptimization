#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <fstream>
#include <map>

using namespace std;
using namespace Eigen;

// Fit a circle to an array of radii using least squares method
tuple<double, double, double> fitCircle(const vector<double>& radii, const vector<double>& angles) {
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


int main() {
    ifstream file("C:/Users/claud/Downloads/outputLog.txt");
    map<double, vector<double>> column_groups_first;
    map<double, vector<double>> column_groups_third;
    vector<double> radius;
    double smallest_radius = 0;

    for (string line; getline(file, line);) {
        istringstream iss(line);
        double first_value, second_value, third_value;
        // group data by rings on the y-axis
        if (iss >> first_value >> second_value >> third_value) {
            column_groups_first[second_value].push_back(first_value);
            column_groups_third[second_value].push_back(third_value);
        }
    }

    for (const auto& pair : column_groups_first) {
        double key = pair.first;
        vector<double> radii = pair.second;
        vector<double> angles = column_groups_third[key];

        auto circleParams = fitCircle(radii, angles);
        cout << "Circle Center (x0, y0): " << get<0>(circleParams) << ", " << get<1>(circleParams) << endl;
        cout << "Circle Radius (r): " << get<2>(circleParams) << endl;
        radius.push_back(get<2>(circleParams));
    }

    smallest_radius = *min_element(radius.begin(), radius.end());
    cout << "Smallest radius: " << smallest_radius << endl;

    return 0;
}