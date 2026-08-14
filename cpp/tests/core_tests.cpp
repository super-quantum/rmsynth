#include "rmsynth_reference/core.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using rmsynth_reference::cnot;

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string{message});
    }
}

template <typename Exception, typename Function>
void require_throws(Function function, const std::string_view message) {
    try {
        function();
    } catch (const Exception &) {
        return;
    }
    throw std::runtime_error(std::string{message});
}

unsigned rank(std::vector<std::uint32_t> rows) {
    unsigned result = 0;
    for (std::size_t column = 0; column < rows.size(); ++column) {
        auto pivot = static_cast<std::size_t>(result);
        while (pivot < rows.size() && ((rows[pivot] >> column) & 1U) == 0U) {
            ++pivot;
        }
        if (pivot == rows.size()) {
            continue;
        }
        std::swap(rows[result], rows[pivot]);
        for (auto row = static_cast<std::size_t>(result); ++row < rows.size();) {
            if (((rows[row] >> column) & 1U) != 0U) {
                rows[row] ^= rows[result];
            }
        }
        ++result;
    }
    return result;
}

std::vector<std::uint32_t> apply_gates(const unsigned qubits, const std::vector<cnot> &gates) {
    std::vector<std::uint32_t> rows;
    for (unsigned index = 0; index < qubits; ++index) {
        rows.push_back(1U << index);
    }
    for (const auto &[control, target] : gates) {
        rows[target] ^= rows[control];
    }
    return rows;
}

void test_reed_muller() {
    using namespace rmsynth_reference;
    require(rm_dimension(4, -1) == 0, "RM(-1,4) dimension");
    require(rm_dimension(4, 0) == 1, "RM(0,4) dimension");
    require(rm_dimension(5, 1) == 6, "RM(1,5) dimension");
    require(rm_basis_terms(5, 1) == std::vector<std::uint32_t>({0, 1, 2, 4, 8, 16}),
            "RM(1,5) terms");
    require(rm_generator_rows(4, 0) == std::vector<std::uint32_t>({0x7FFF}), "RM(0,4) generator");
    require_throws<validation_error>([] { rm_dimension(0, 0); }, "zero qubits accepted");
    require_throws<validation_error>([] { rm_dimension(5, 5); }, "full order accepted");
}

void test_decoder() {
    using namespace rmsynth_reference;
    const auto zero = decode_exact(0, 4, 0);
    require(zero.codeword == 0 && zero.distance == 0 && zero.ties == 1, "zero decode");

    const auto ones = decode_exact(0x7FFF, 4, 0);
    require(ones.codeword == 0x7FFF && ones.distance == 0, "all-ones decode");

    const auto tie = decode_exact(33023, 5, 1);
    require(tie.codeword == 32767 && tie.distance == 8 && tie.ties == 2, "tie decode");
    require(tie.selected_terms == std::vector<std::uint32_t>({0, 16}), "tie selection");

    const auto maximum = decode_exact(0, 5, 2);
    require(maximum.codeword == 0 && maximum.distance == 0 &&
                maximum.candidates == max_decoder_candidates,
            "maximum bounded decode");

    require_throws<validation_error>([] { decode_exact(1U << 15U, 4, 0); },
                                     "oversized received word accepted");
    require_throws<resource_limit_error>([] { decode_exact(0, 5, 4); },
                                         "candidate limit not enforced");
}

void test_linear_maps() {
    using namespace rmsynth_reference;
    require(synthesize_linear_map({1, 2, 4}).empty(), "identity synthesis");
    require(apply_gates(2, synthesize_linear_map({2, 1})) == std::vector<std::uint32_t>({2, 1}),
            "swap synthesis");

    unsigned invertible = 0;
    for (std::uint32_t packed = 0; packed < (1U << 9U); ++packed) {
        const std::vector<std::uint32_t> rows = {
            packed & 7U,
            (packed >> 3U) & 7U,
            (packed >> 6U) & 7U,
        };
        if (rank(rows) != 3) {
            require_throws<validation_error>([&rows] { synthesize_linear_map(rows); },
                                             "singular map accepted");
            continue;
        }
        ++invertible;
        require(apply_gates(3, synthesize_linear_map(rows)) == rows, "linear-map synthesis");
    }
    require(invertible == 168, "invertible 3x3 map count");
    require_throws<validation_error>([] { synthesize_linear_map({}); }, "empty map accepted");
    require_throws<validation_error>([] { synthesize_linear_map({2}); }, "wide row accepted");
}

} // namespace

int main() {
    try {
        test_reed_muller();
        test_decoder();
        test_linear_maps();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
