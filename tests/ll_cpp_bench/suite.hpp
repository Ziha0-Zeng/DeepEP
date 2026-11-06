#pragma once
#include "options.hpp"
#include <filesystem>

namespace llbench {
struct Suite {
    std::vector<std::string> variants, arguments;
    std::filesystem::path buildRoot, executableDirectory;
    bool explicitVariants = false;
    Suite(int argc, char** argv, const std::filesystem::path& executable)
        : buildRoot(executable.parent_path().parent_path()), executableDirectory(executable.parent_path())
    {
        arguments.emplace_back(argv[0]);
        bool haveRoot = false;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc) throw std::invalid_argument("Expected --option value pairs");
            const std::string key = argv[i], value = argv[i + 1];
            if (key == "--variants") {
                if (explicitVariants || value.empty() || value.back() == ',')
                    throw std::invalid_argument("Provide one nonempty --variants list");
                explicitVariants = true;
                std::stringstream names(value);
                std::string name;
                while (std::getline(names, name, ',')) {
                    const std::string allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-";
                    if (name.empty() || name.find_first_not_of(allowed) != std::string::npos ||
                        name.find_first_of("_-") == 0)
                        throw std::invalid_argument("Invalid variant name: " + name);
                    variants.push_back(name);
                }
                if (variants.size() > 8) throw std::invalid_argument("At most eight phases are supported");
            } else if (key == "--build-root") {
                if (haveRoot || value.empty()) throw std::invalid_argument("Invalid --build-root");
                haveRoot = true;
                buildRoot = value;
            } else {
                arguments.push_back(key);
                arguments.push_back(value);
            }
        }
        if (!explicitVariants) variants.push_back(executableDirectory.filename().string());
    }
    Options options(uint32_t rank, uint32_t world, uint32_t localRank)
    {
        std::vector<char*> argv;
        for (auto& arg : arguments) argv.push_back(arg.data());
        return Options::parse(int(argv.size()), argv.data(), rank, world, localRank);
    }
    std::filesystem::path library(size_t i) const
    {
        return (explicitVariants ? buildRoot / ("ll-cpp-" + variants.at(i)) : executableDirectory) /
               "libll_bench_variant.so";
    }
    std::filesystem::path output(const std::string& root, size_t i) const
    {
        return explicitVariants ? std::filesystem::path(root) / (std::to_string(i) + "-" + variants.at(i))
                                : std::filesystem::path(root);
    }
    std::string signature() const
    {
        std::string result;
        for (const auto& name : variants) result += "/" + name;
        return result;
    }
};
}  // namespace llbench
