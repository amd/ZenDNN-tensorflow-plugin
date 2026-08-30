// Simple std::regex helper for AMD plugin
#pragma once

#include <string>
#include <vector>

namespace zendnn_tf_plugin {

// Finds all non-overlapping matches of `pattern` in `text`.
// Returns true on success; `out_matches` will be filled with each match substring.
bool regex_find_all(const std::string &text, const std::string &pattern,
                    std::vector<std::string> &out_matches);

}  // namespace zendnn_tf_plugin
