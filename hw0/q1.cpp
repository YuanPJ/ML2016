#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

double getData(string str, int col){
  size_t start = 0, end = 0, length = 0;
  for (int count = -1; count < col; count++){
    start = str.find_first_of(' ', start);
    ++start;
  }
  str = str.substr(start);
  return stod(str);
}

int main(int argc, char** argv)
{
  int col = atoi(argv[1]);
  fstream inf;
  fstream outf;

  inf.open(argv[2], fstream::in);
  outf.open("ans1.txt", fstream::out);

  vector<double> ans;
  string str;
  while (getline(inf, str))
    ans.push_back(getData(str, col));
  inf.close();

  sort(ans.begin(), ans.end());
  outf << ans[0];
  for (size_t i = 1; i < ans.size(); i++)
    outf << ',' << ans[i];

}
