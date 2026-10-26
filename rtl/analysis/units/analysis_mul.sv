// Functional-only measurement boundary; debug outputs are intentionally pruned.
module analysis_mul (input logic [15:0] a, b, output logic [15:0] result, output logic [3:0] flags);
    logic [10:0] unused_significand_a;
    logic [10:0] unused_significand_b;
    logic unused_product_sign;
    logic signed [7:0] unused_product_exponent;
    logic [21:0] unused_exact_product;
    logic [79:0] unused_aligned_product;
    logic [63:0] unused_aligned_addend;
    logic [80:0] unused_arithmetic_magnitude;
    logic unused_result_sign;
    logic unused_special_case;
    logic [6:0] unused_normalization_shift;
    logic [6:0] unused_discard_count;
    logic [11:0] unused_retained_significand;
    logic unused_guard_bit;
    logic unused_round_bit;
    logic unused_sticky_bit;
    logic unused_increment;
    logic [5:0] unused_exponent_rounded;
    logic unused_tiny_after_precision_rounding;
    fp16_mul dut (
        .a(a),
        .b(b),
        .result(result),
        .invalid(flags[3]),
        .overflow(flags[2]),
        .underflow(flags[1]),
        .inexact(flags[0]),
        .significand_a(unused_significand_a),
        .significand_b(unused_significand_b),
        .product_sign(unused_product_sign),
        .product_exponent(unused_product_exponent),
        .exact_product(unused_exact_product),
        .aligned_product(unused_aligned_product),
        .aligned_addend(unused_aligned_addend),
        .arithmetic_magnitude(unused_arithmetic_magnitude),
        .result_sign(unused_result_sign),
        .special_case(unused_special_case),
        .normalization_shift(unused_normalization_shift),
        .discard_count(unused_discard_count),
        .retained_significand(unused_retained_significand),
        .guard_bit(unused_guard_bit),
        .round_bit(unused_round_bit),
        .sticky_bit(unused_sticky_bit),
        .increment(unused_increment),
        .exponent_rounded(unused_exponent_rounded),
        .tiny_after_precision_rounding(unused_tiny_after_precision_rounding)
    );
endmodule
