// Functional-only measurement boundary; debug outputs are intentionally pruned.
module analysis_add (input logic [15:0] a, b, input logic subtract, output logic [15:0] result, output logic [3:0] flags);
    logic unused_effective_b_sign;
    logic [5:0] unused_exponent_difference;
    logic [39:0] unused_aligned_a;
    logic [39:0] unused_aligned_b;
    logic [40:0] unused_arithmetic_magnitude;
    logic [5:0] unused_normalization_shift;
    logic [40:0] unused_normalized_magnitude;
    logic [5:0] unused_discard_count;
    logic [11:0] unused_retained_significand;
    logic unused_guard_bit;
    logic unused_round_bit;
    logic unused_sticky_bit;
    logic unused_increment;
    logic [4:0] unused_exponent_rounded;
    logic unused_special_case;
    fp16_addsub dut (
        .a(a),
        .b(b),
        .subtract(subtract),
        .result(result),
        .invalid(flags[3]),
        .overflow(flags[2]),
        .underflow(flags[1]),
        .inexact(flags[0]),
        .effective_b_sign(unused_effective_b_sign),
        .exponent_difference(unused_exponent_difference),
        .aligned_a(unused_aligned_a),
        .aligned_b(unused_aligned_b),
        .arithmetic_magnitude(unused_arithmetic_magnitude),
        .normalization_shift(unused_normalization_shift),
        .normalized_magnitude(unused_normalized_magnitude),
        .discard_count(unused_discard_count),
        .retained_significand(unused_retained_significand),
        .guard_bit(unused_guard_bit),
        .round_bit(unused_round_bit),
        .sticky_bit(unused_sticky_bit),
        .increment(unused_increment),
        .exponent_rounded(unused_exponent_rounded),
        .special_case(unused_special_case)
    );
endmodule
