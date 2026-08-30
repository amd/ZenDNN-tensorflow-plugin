#include <iostream>
#include "regex_lib.h"

int main() {
  std::string text = "abc 123 def 456 78";
  std::string pattern = "\\d+";  // find numbers
  std::vector<std::string> matches;
  if (!zendnn_tf_plugin::regex_find_all(text, pattern, matches)) {
    std::cerr << "regex error" << std::endl;
    return 2;
  }
  for (auto &m : matches) std::cout << m << std::endl;
  return matches.empty() ? 1 : 0;
}
