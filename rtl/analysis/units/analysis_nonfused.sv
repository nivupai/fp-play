// Functional-only measurement boundary; debug outputs are intentionally pruned.
module analysis_nonfused (input logic [15:0] a, b, c, output logic [15:0] result, output logic [3:0] flags);
    logic [15:0] unused_multiply_result;
    logic [15:0] unused_fused_result;
    logic [3:0] unused_multiply_flags;
    logic [3:0] unused_fused_flags;
    logic [3:0] unused_addition_flags;
    logic [10:0] unused_mul_significand_a;
    logic [10:0] unused_mul_significand_b;
    logic unused_mul_product_sign;
    logic signed [7:0] unused_mul_product_exponent;
    logic [21:0] unused_mul_exact_product;
    logic [79:0] unused_mul_aligned_product;
    logic [63:0] unused_mul_aligned_addend;
    logic [80:0] unused_mul_arithmetic_magnitude;
    logic unused_mul_result_sign;
    logic unused_mul_special_case;
    logic [6:0] unused_mul_normalization_shift;
    logic [6:0] unused_mul_discard_count;
    logic [11:0] unused_mul_retained_significand;
    logic unused_mul_guard_bit;
    logic unused_mul_round_bit;
    logic unused_mul_sticky_bit;
    logic unused_mul_increment;
    logic [5:0] unused_mul_exponent_rounded;
    logic unused_mul_tiny_after_precision_rounding;
    logic [10:0] unused_fma_significand_a;
    logic [10:0] unused_fma_significand_b;
    logic unused_fma_product_sign;
    logic signed [7:0] unused_fma_product_exponent;
    logic [21:0] unused_fma_exact_product;
    logic [79:0] unused_fma_aligned_product;
    logic [63:0] unused_fma_aligned_addend;
    logic [80:0] unused_fma_arithmetic_magnitude;
    logic unused_fma_result_sign;
    logic unused_fma_special_case;
    logic [6:0] unused_fma_normalization_shift;
    logic [6:0] unused_fma_discard_count;
    logic [11:0] unused_fma_retained_significand;
    logic unused_fma_guard_bit;
    logic unused_fma_round_bit;
    logic unused_fma_sticky_bit;
    logic unused_fma_increment;
    logic [5:0] unused_fma_exponent_rounded;
    logic unused_fma_tiny_after_precision_rounding;
    logic unused_add_effective_b_sign;
    logic [5:0] unused_add_exponent_difference;
    logic [39:0] unused_add_aligned_a;
    logic [39:0] unused_add_aligned_b;
    logic [40:0] unused_add_arithmetic_magnitude;
    logic [5:0] unused_add_normalization_shift;
    logic [40:0] unused_add_normalized_magnitude;
    logic [5:0] unused_add_discard_count;
    logic [11:0] unused_add_retained_significand;
    logic unused_add_guard_bit;
    logic unused_add_round_bit;
    logic unused_add_sticky_bit;
    logic unused_add_increment;
    logic [4:0] unused_add_exponent_rounded;
    logic unused_add_special_case;
    fp16_mul_fma_compare dut (
        .a(a),
        .b(b),
        .c(c),
        .multiply_result(unused_multiply_result),
        .fused_result(unused_fused_result),
        .nonfused_result(result),
        .multiply_flags(unused_multiply_flags),
        .fused_flags(unused_fused_flags),
        .addition_flags(unused_addition_flags),
        .nonfused_flags(flags),
        .mul_significand_a(unused_mul_significand_a),
        .mul_significand_b(unused_mul_significand_b),
        .mul_product_sign(unused_mul_product_sign),
        .mul_product_exponent(unused_mul_product_exponent),
        .mul_exact_product(unused_mul_exact_product),
        .mul_aligned_product(unused_mul_aligned_product),
        .mul_aligned_addend(unused_mul_aligned_addend),
        .mul_arithmetic_magnitude(unused_mul_arithmetic_magnitude),
        .mul_result_sign(unused_mul_result_sign),
        .mul_special_case(unused_mul_special_case),
        .mul_normalization_shift(unused_mul_normalization_shift),
        .mul_discard_count(unused_mul_discard_count),
        .mul_retained_significand(unused_mul_retained_significand),
        .mul_guard_bit(unused_mul_guard_bit),
        .mul_round_bit(unused_mul_round_bit),
        .mul_sticky_bit(unused_mul_sticky_bit),
        .mul_increment(unused_mul_increment),
        .mul_exponent_rounded(unused_mul_exponent_rounded),
        .mul_tiny_after_precision_rounding(unused_mul_tiny_after_precision_rounding),
        .fma_significand_a(unused_fma_significand_a),
        .fma_significand_b(unused_fma_significand_b),
        .fma_product_sign(unused_fma_product_sign),
        .fma_product_exponent(unused_fma_product_exponent),
        .fma_exact_product(unused_fma_exact_product),
        .fma_aligned_product(unused_fma_aligned_product),
        .fma_aligned_addend(unused_fma_aligned_addend),
        .fma_arithmetic_magnitude(unused_fma_arithmetic_magnitude),
        .fma_result_sign(unused_fma_result_sign),
        .fma_special_case(unused_fma_special_case),
        .fma_normalization_shift(unused_fma_normalization_shift),
        .fma_discard_count(unused_fma_discard_count),
        .fma_retained_significand(unused_fma_retained_significand),
        .fma_guard_bit(unused_fma_guard_bit),
        .fma_round_bit(unused_fma_round_bit),
        .fma_sticky_bit(unused_fma_sticky_bit),
        .fma_increment(unused_fma_increment),
        .fma_exponent_rounded(unused_fma_exponent_rounded),
        .fma_tiny_after_precision_rounding(unused_fma_tiny_after_precision_rounding),
        .add_effective_b_sign(unused_add_effective_b_sign),
        .add_exponent_difference(unused_add_exponent_difference),
        .add_aligned_a(unused_add_aligned_a),
        .add_aligned_b(unused_add_aligned_b),
        .add_arithmetic_magnitude(unused_add_arithmetic_magnitude),
        .add_normalization_shift(unused_add_normalization_shift),
        .add_normalized_magnitude(unused_add_normalized_magnitude),
        .add_discard_count(unused_add_discard_count),
        .add_retained_significand(unused_add_retained_significand),
        .add_guard_bit(unused_add_guard_bit),
        .add_round_bit(unused_add_round_bit),
        .add_sticky_bit(unused_add_sticky_bit),
        .add_increment(unused_add_increment),
        .add_exponent_rounded(unused_add_exponent_rounded),
        .add_special_case(unused_add_special_case)
    );
endmodule
