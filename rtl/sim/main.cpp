// Original file-driven differential simulator; expected results come from TypeScript.
#include "Vfp16_addsub.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: simulator vectors.txt observations.txt [wave.vcd]\n";
        return 2;
    }
    auto context = std::make_unique<VerilatedContext>();
    context->commandArgs(argc, argv);
    context->traceEverOn(argc == 4);
    auto dut = std::make_unique<Vfp16_addsub>(context.get());
    auto wave = std::make_unique<VerilatedVcdC>();
    if (argc == 4) {
        dut->trace(wave.get(), 99);
        wave->open(argv[3]);
    }
    std::ifstream input(argv[1]);
    std::ofstream observations(argv[2]);
    if (!input || !observations) {
        std::cerr << "Cannot open vector/output file\n";
        return 2;
    }
    std::string line;
    std::uint64_t cases = 0, mismatches = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream row(line);
        unsigned a, b, subtract, expected, flags;
        if (!(row >> std::hex >> a >> b >> subtract >> expected >> flags) || a > 0xffff ||
            b > 0xffff || subtract > 1 || expected > 0xffff || flags > 15) {
            std::cerr << "Invalid vector line " << cases + 1 << '\n';
            return 2;
        }
        // Record a settled baseline and then a settled transaction. Timestamps are
        // observation indices, NOT a hardware clock or propagation-delay model.
        if (argc == 4) {
            dut->a = 0;
            dut->b = 0;
            dut->subtract = 0;
            dut->eval();
            wave->dump(context->time());
            context->timeInc(1);
        }
        dut->a = a;
        dut->b = b;
        dut->subtract = subtract;
        dut->eval();
        if (argc == 4) {
            wave->dump(context->time());
            context->timeInc(1);
        }
        const unsigned actualFlags =
            (dut->invalid << 3) | (dut->overflow << 2) | (dut->underflow << 1) | dut->inexact;
        const bool match = dut->result == expected && actualFlags == flags;
        // Strict raw comparison includes NaN payload and sign; no exemptions.
        if (!match) {
            mismatches++;
            if (mismatches <= 10)
                std::cerr << "Mismatch " << cases << ": " << line << " actual " << std::hex
                          << dut->result << '/' << actualFlags << std::dec << '\n';
        }
        observations << std::hex << a << ' ' << b << ' ' << subtract << ' ' << dut->result << ' '
                     << actualFlags << ' ' << unsigned(dut->effective_b_sign) << ' '
                     << unsigned(dut->exponent_difference) << ' ' << dut->aligned_a << ' '
                     << dut->aligned_b << ' ' << dut->arithmetic_magnitude << ' '
                     << unsigned(dut->normalization_shift) << ' ' << dut->normalized_magnitude
                     << ' ' << unsigned(dut->discard_count) << ' '
                     << unsigned(dut->retained_significand) << ' ' << unsigned(dut->guard_bit)
                     << ' ' << unsigned(dut->round_bit) << ' ' << unsigned(dut->sticky_bit) << ' '
                     << unsigned(dut->increment) << ' ' << unsigned(dut->exponent_rounded) << ' '
                     << unsigned(dut->special_case) << '\n';
        cases++;
    }
    dut->final();
    if (argc == 4)
        wave->close();
    if (!cases) {
        std::cerr << "No test vectors\n";
        return 2;
    }
    std::cout << "RTL cases=" << cases << " mismatches=" << mismatches << '\n';
    return mismatches ? 1 : 0;
}
