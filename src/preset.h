#pragma once
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <sys/stat.h>

inline void savePreset(const std::string& filename,
                       const std::map<std::string, std::vector<float>>& data) {
    std::string dir = "presets";
    mkdir(dir.c_str(), 0755);
    std::string path = dir + "/" + filename + ".txt";
    std::ofstream f(path);
    for (auto& [key, vals] : data) {
        f << key;
        for (float v : vals) f << " " << v;
        f << "\n";
    }
}

inline std::map<std::string, std::vector<float>> loadPreset(const std::string& filename) {
    std::map<std::string, std::vector<float>> data;
    std::string path = "presets/" + filename + ".txt";
    std::ifstream f(path);
    if (!f.is_open()) return data;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        std::vector<float> vals;
        float v;
        while (ss >> v) vals.push_back(v);
        if (!key.empty()) data[key] = vals;
    }
    return data;
}
