#include "rmsynth_reference/core.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <string>

namespace rmsynth_reference {
namespace {

void validate_rm_parameters(const unsigned qubits, const int order) {
    if (qubits == 0 || qubits > max_qubits) {
        throw validation_error("qubits must be between 1 and 5");
    }
    if (order < -1 || order >= static_cast<int>(qubits)) {
        throw validation_error(
            "punctured Reed-Muller order must be between -1 and qubits minus one");
    }
}

unsigned rank(std::vector<std::uint32_t> rows) {
    unsigned result = 0;
    for (unsigned column = 0; column < rows.size(); ++column) {
        const auto pivot = std::find_if(
            rows.begin() + static_cast<std::ptrdiff_t>(result), rows.end(),
            [column](const std::uint32_t row) { return ((row >> column) & 1U) != 0U; });
        if (pivot == rows.end()) {
            continue;
        }
        std::iter_swap(rows.begin() + static_cast<std::ptrdiff_t>(result), pivot);
        for (unsigned row = 0; row < rows.size(); ++row) {
            if (row != result && ((rows[row] >> column) & 1U) != 0U) {
                rows[row] ^= rows[result];
            }
        }
        ++result;
    }
    return result;
}

} // namespace

unsigned rm_dimension(const unsigned qubits, const int order) {
    validate_rm_parameters(qubits, order);
    if (order < 0) {
        return 0;
    }

    unsigned dimension = 0;
    unsigned binomial = 1;
    for (int degree = 0; degree <= order; ++degree) {
        if (degree > 0) {
            const auto numerator = qubits - static_cast<unsigned>(degree) + 1U;
            binomial = binomial * numerator / static_cast<unsigned>(degree);
        }
        dimension += binomial;
    }
    return dimension;
}

std::vector<std::uint32_t> rm_basis_terms(const unsigned qubits, const int order) {
    validate_rm_parameters(qubits, order);
    std::vector<std::uint32_t> terms;
    if (order < 0) {
        return terms;
    }

    terms.reserve(rm_dimension(qubits, order));
    terms.push_back(0);
    for (std::uint32_t mask = 1; mask < (1U << qubits); ++mask) {
        if (std::popcount(mask) <= order) {
            terms.push_back(mask);
        }
    }
    return terms;
}

std::vector<std::uint32_t> rm_generator_rows(const unsigned qubits, const int order) {
    const auto terms = rm_basis_terms(qubits, order);
    std::vector<std::uint32_t> rows;
    rows.reserve(terms.size());
    for (const auto term : terms) {
        std::uint32_t row = 0;
        for (std::uint32_t point = 1; point < (1U << qubits); ++point) {
            if (term == 0 || (term & point) == term) {
                row |= 1U << (point - 1U);
            }
        }
        rows.push_back(row);
    }
    return rows;
}

decode_result decode_exact(const std::uint64_t received, const unsigned qubits, const int order) {
    const auto dimension = rm_dimension(qubits, order);
    const auto candidates = std::uint64_t{1} << dimension;
    if (candidates > max_decoder_candidates) {
        throw resource_limit_error("exact decoding needs " + std::to_string(candidates) +
                                   " candidates; limit is " +
                                   std::to_string(max_decoder_candidates));
    }

    const auto length = (1U << qubits) - 1U;
    const auto word_limit = std::uint64_t{1} << length;
    if (received >= word_limit) {
        throw validation_error("received word exceeds the punctured code length");
    }

    const auto rows = rm_generator_rows(qubits, order);
    std::uint32_t current_word = 0;
    std::uint64_t previous_gray = 0;
    std::uint64_t best_selection = 0;
    decode_result result{0, {}, static_cast<unsigned>(std::popcount(received)), candidates, 1};

    for (std::uint64_t index = 1; index < candidates; ++index) {
        const auto gray = index ^ (index >> 1U);
        const auto changed = static_cast<std::size_t>(std::countr_zero(gray ^ previous_gray));
        current_word ^= rows[changed];
        const auto distance = static_cast<unsigned>(std::popcount(received ^ current_word));
        if (distance < result.distance) {
            result.codeword = current_word;
            result.distance = distance;
            result.ties = 1;
            best_selection = gray;
        } else if (distance == result.distance) {
            ++result.ties;
            if (current_word < result.codeword) {
                result.codeword = current_word;
                best_selection = gray;
            }
        }
        previous_gray = gray;
    }

    const auto terms = rm_basis_terms(qubits, order);
    for (std::size_t index = 0; index < terms.size(); ++index) {
        if (((best_selection >> index) & 1U) != 0U) {
            result.selected_terms.push_back(terms[index]);
        }
    }
    return result;
}

std::vector<cnot> synthesize_linear_map(const std::vector<std::uint32_t> &input_rows) {
    if (input_rows.empty() || input_rows.size() > max_qubits) {
        throw validation_error("linear map must contain between 1 and 5 rows");
    }
    const auto qubits = static_cast<unsigned>(input_rows.size());
    const auto row_limit = 1U << qubits;
    if (std::any_of(input_rows.begin(), input_rows.end(),
                    [row_limit](const auto row) { return row >= row_limit; })) {
        throw validation_error("linear-map row exceeds its dimension");
    }
    if (rank(input_rows) != qubits) {
        throw validation_error("linear map must be invertible over GF(2)");
    }

    auto rows = input_rows;
    std::vector<cnot> reductions;
    for (unsigned column = 0; column < qubits; ++column) {
        auto pivot = column;
        while (((rows[pivot] >> column) & 1U) == 0U) {
            ++pivot;
        }
        if (pivot != column) {
            for (const auto &gate :
                 {cnot{pivot, column}, cnot{column, pivot}, cnot{pivot, column}}) {
                rows[gate.second] ^= rows[gate.first];
                reductions.push_back(gate);
            }
        }
        for (unsigned row = 0; row < qubits; ++row) {
            if (row != column && ((rows[row] >> column) & 1U) != 0U) {
                rows[row] ^= rows[column];
                reductions.emplace_back(column, row);
            }
        }
    }
    std::reverse(reductions.begin(), reductions.end());
    return reductions;
}

} // namespace rmsynth_reference
