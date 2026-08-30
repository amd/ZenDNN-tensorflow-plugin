#include "regex_lib.h"
#include <regex>

namespace zendnn_tf_plugin {

bool regex_find_all(const std::string &text, const std::string &pattern,
                    std::vector<std::string> &out_matches) {
  out_matches.clear();
  try {
    std::regex re(pattern);
    std::sregex_iterator it(text.begin(), text.end(), re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
      out_matches.push_back((*it).str());
    }
    return true;
  } catch (const std::regex_error &) {
    return false;
  }
}

}  // namespace zendnn_tf_plugin
