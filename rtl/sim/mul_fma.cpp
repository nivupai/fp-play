// Original file-driven RTL comparison. All numerical expectations come from TypeScript.
#include "Vfp16_mul_fma_compare.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

static std::string wideHex(const VlWide<3> &value) {
    std::ostringstream out;
    out << std::hex << value[2] << std::setw(8) << std::setfill('0') << value[1] << std::setw(8)
        << value[0];
    return out.str();
}
int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: simulator vectors.txt observations.txt [wave.vcd]\n";
        return 2;
    }
    auto context = std::make_unique<VerilatedContext>();
    context->commandArgs(argc, argv);
    context->traceEverOn(argc == 4);
    auto dut = std::make_unique<Vfp16_mul_fma_compare>(context.get());
    auto wave = std::make_unique<VerilatedVcdC>();
    if (argc == 4) {
        dut->trace(wave.get(), 99);
        wave->open(argv[3]);
    }
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) {
        std::cerr << "Cannot open vectors/observations\n";
        return 2;
    }
    std::string line;
    unsigned long long cases = 0, mismatches = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        unsigned a, b, c, m, mf, f, ff, n, nf, af;
        std::istringstream row(line);
        if (!(row >> std::hex >> a >> b >> c >> m >> mf >> f >> ff >> n >> nf >> af) ||
            a > 0xffff || b > 0xffff || c > 0xffff || m > 0xffff || f > 0xffff || n > 0xffff ||
            mf > 15 || ff > 15 || nf > 15 || af > 15) {
            std::cerr << "Malformed vector\n";
            return 2;
        }
        std::string extra;
        if (row >> extra) {
            std::cerr << "Unexpected trailing vector field\n";
            return 2;
        }
        if (argc == 4) {
            dut->a = 0;
            dut->b = 0;
            dut->c = 0;
            dut->eval();
            wave->dump(context->time());
            context->timeInc(1);
        }
        dut->a = a;
        dut->b = b;
        dut->c = c;
        dut->eval();
        if (argc == 4) {
            wave->dump(context->time());
            context->timeInc(1);
        }
        if (dut->multiply_result != m || dut->multiply_flags != mf || dut->fused_result != f ||
            dut->fused_flags != ff || dut->nonfused_result != n || dut->nonfused_flags != nf ||
            dut->addition_flags != af) {
            mismatches++;
            if (mismatches <= 8)
                std::cerr << "Mismatch " << cases << ": " << line << '\n';
        }
        output << std::hex << a << ' ' << b << ' ' << c << ' ' << dut->multiply_result << ' '
               << unsigned(dut->multiply_flags) << ' ' << dut->fused_result << ' '
               << unsigned(dut->fused_flags) << ' ' << dut->nonfused_result << ' '
               << unsigned(dut->nonfused_flags) << ' ' << unsigned(dut->addition_flags);
        output << ' ' << unsigned(dut->mul_significand_a);
        output << ' ' << unsigned(dut->mul_significand_b);
        output << ' ' << unsigned(dut->mul_product_sign);
        output << ' ' << unsigned(dut->mul_product_exponent);
        output << ' ' << unsigned(dut->mul_exact_product);
        output << ' ' << wideHex(dut->mul_aligned_product);
        output << ' ' << dut->mul_aligned_addend;
        output << ' ' << wideHex(dut->mul_arithmetic_magnitude);
        output << ' ' << unsigned(dut->mul_result_sign);
        output << ' ' << unsigned(dut->mul_special_case);
        output << ' ' << unsigned(dut->mul_normalization_shift);
        output << ' ' << unsigned(dut->mul_discard_count);
        output << ' ' << unsigned(dut->mul_retained_significand);
        output << ' ' << unsigned(dut->mul_guard_bit);
        output << ' ' << unsigned(dut->mul_round_bit);
        output << ' ' << unsigned(dut->mul_sticky_bit);
        output << ' ' << unsigned(dut->mul_increment);
        output << ' ' << unsigned(dut->mul_exponent_rounded);
        output << ' ' << unsigned(dut->mul_tiny_after_precision_rounding);
        output << ' ' << unsigned(dut->fma_significand_a);
        output << ' ' << unsigned(dut->fma_significand_b);
        output << ' ' << unsigned(dut->fma_product_sign);
        output << ' ' << unsigned(dut->fma_product_exponent);
        output << ' ' << unsigned(dut->fma_exact_product);
        output << ' ' << wideHex(dut->fma_aligned_product);
        output << ' ' << dut->fma_aligned_addend;
        output << ' ' << wideHex(dut->fma_arithmetic_magnitude);
        output << ' ' << unsigned(dut->fma_result_sign);
        output << ' ' << unsigned(dut->fma_special_case);
        output << ' ' << unsigned(dut->fma_normalization_shift);
        output << ' ' << unsigned(dut->fma_discard_count);
        output << ' ' << unsigned(dut->fma_retained_significand);
        output << ' ' << unsigned(dut->fma_guard_bit);
        output << ' ' << unsigned(dut->fma_round_bit);
        output << ' ' << unsigned(dut->fma_sticky_bit);
        output << ' ' << unsigned(dut->fma_increment);
        output << ' ' << unsigned(dut->fma_exponent_rounded);
        output << ' ' << unsigned(dut->fma_tiny_after_precision_rounding);
        output << '\n';
        cases++;
    }
    dut->final();
    if (argc == 4)
        wave->close();
    if (!cases) {
        std::cerr << "No vectors\n";
        return 2;
    }
    std::cout << "RTL multiply/FMA triples=" << cases << " mismatches=" << mismatches << '\n';
    return mismatches ? 1 : 0;
}
