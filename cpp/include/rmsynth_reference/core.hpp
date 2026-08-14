#pragma once

#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rmsynth_reference {

inline constexpr unsigned max_qubits = 5;
inline constexpr std::uint64_t max_decoder_candidates = 65'536;

class validation_error : public std::invalid_argument {
  public:
    using std::invalid_argument::invalid_argument;
};

class resource_limit_error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

struct decode_result {
    std::uint32_t codeword{};
    std::vector<std::uint32_t> selected_terms;
    unsigned distance{};
    std::uint64_t candidates{};
    std::uint64_t ties{};
};

using cnot = std::pair<unsigned, unsigned>;

unsigned rm_dimension(unsigned qubits, int order);
std::vector<std::uint32_t> rm_basis_terms(unsigned qubits, int order);
std::vector<std::uint32_t> rm_generator_rows(unsigned qubits, int order);
decode_result decode_exact(std::uint64_t received, unsigned qubits, int order);
std::vector<cnot> synthesize_linear_map(const std::vector<std::uint32_t> &rows);

} // namespace rmsynth_reference
