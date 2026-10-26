// Original exact-product core: include_c=0 multiplies; include_c=1 performs true FMA.
// Nothing rounds or range-limits the product before the exact addition.
module fp16_mul_fma_core (
    input logic [15:0] a, b, c,
    input logic include_c,
    output logic [15:0] result,
    output logic invalid,
    output logic overflow,
    output logic underflow,
    output logic inexact,
    output logic [10:0] significand_a,
    output logic [10:0] significand_b,
    output logic product_sign,
    output logic signed [7:0] product_exponent,
    output logic [21:0] exact_product,
    output logic [79:0] aligned_product,
    output logic [63:0] aligned_addend,
    output logic [80:0] arithmetic_magnitude,
    output logic result_sign,
    output logic special_case,
    output logic [6:0] normalization_shift,
    output logic [6:0] discard_count,
    output logic [11:0] retained_significand,
    output logic guard_bit,
    output logic round_bit,
    output logic sticky_bit,
    output logic increment,
    output logic [5:0] exponent_rounded,
    output logic tiny_after_precision_rounding
);
    logic a_nan, b_nan, c_nan, a_snan, b_snan, c_snan;
    logic a_inf, b_inf, c_inf, invalid_product, infinite_product;
    logic [4:0] ea, eb, ec;
    logic [5:0] product_shift;
    logic [10:0] significand_c;
    logic [15:0] finite_result;
    logic finite_overflow, finite_underflow, finite_inexact;
    always_comb begin
        a_nan = (&a[14:10]) && (|a[9:0]);
        b_nan = (&b[14:10]) && (|b[9:0]);
        c_nan = include_c && (&c[14:10]) && (|c[9:0]);
        a_snan = a_nan && !a[9];
        b_snan = b_nan && !b[9];
        c_snan = c_nan && !c[9];
        a_inf = (&a[14:10]) && !(|a[9:0]);
        b_inf = (&b[14:10]) && !(|b[9:0]);
        c_inf = include_c && (&c[14:10]) && !(|c[9:0]);
        invalid_product = (a_inf && (b[14:0] == 15'd0)) ||
                          (b_inf && (a[14:0] == 15'd0));
        infinite_product = a_inf || b_inf;
        ea = a[14:10] == 5'd0 ? 5'd1 : a[14:10];
        eb = b[14:10] == 5'd0 ? 5'd1 : b[14:10];
        ec = c[14:10] == 5'd0 ? 5'd1 : c[14:10];
        significand_a = {(a[14:10] != 5'd0), a[9:0]};
        significand_b = {(b[14:10] != 5'd0), b[9:0]};
        significand_c = {(c[14:10] != 5'd0), c[9:0]};
        product_sign = a[15] ^ b[15];
        product_shift = {1'b0, ea} + {1'b0, eb} - 6'd2;
        product_exponent = $signed({3'd0, ea}) + $signed({3'd0, eb}) - 8'sd50;
        special_case = a_nan || b_nan || c_nan || invalid_product || infinite_product || c_inf;
        exact_product = 22'd0;
        aligned_product = 80'd0;
        aligned_addend = 64'd0;
        arithmetic_magnitude = 81'd0;
        result_sign = product_sign;
        if (!special_case) begin
            // Exact 11x11 multiply: 22 bits, then shift by 0..58 on the 2^-48 grid.
            exact_product = significand_a * significand_b;
            aligned_product = {58'd0, exact_product} << product_shift;
            if (include_c) aligned_addend = {53'd0, significand_c} << ({1'b0, ec} + 6'd23);
            if (!include_c) arithmetic_magnitude = {1'b0, aligned_product};
            else if (product_sign == c[15]) begin
                arithmetic_magnitude = {1'b0, aligned_product} + {17'd0, aligned_addend};
                result_sign = product_sign;
            end else if ({1'b0, aligned_product} >= {17'd0, aligned_addend}) begin
                arithmetic_magnitude = {1'b0, aligned_product} - {17'd0, aligned_addend};
                result_sign = ({1'b0, aligned_product} == {17'd0, aligned_addend}) ? 1'b0 : product_sign;
            end else begin
                arithmetic_magnitude = {17'd0, aligned_addend} - {1'b0, aligned_product};
                result_sign = c[15];
            end
        end
    end
    fp16_round_grid48 rounder (
        .magnitude(arithmetic_magnitude), .sign(result_sign), .result(finite_result),
        .overflow(finite_overflow), .underflow(finite_underflow), .inexact(finite_inexact),
        .normalization_shift(normalization_shift),
        .discard_count(discard_count),
        .retained_significand(retained_significand),
        .guard_bit(guard_bit),
        .round_bit(round_bit),
        .sticky_bit(sticky_bit),
        .increment(increment),
        .exponent_rounded(exponent_rounded),
        .tiny_after_precision_rounding(tiny_after_precision_rounding)
    );
    always_comb begin
        result = finite_result;
        invalid = a_snan || b_snan || c_snan || invalid_product;
        overflow = finite_overflow;
        underflow = finite_underflow;
        inexact = finite_inexact;
        if (special_case) begin
            overflow = 1'b0;
            underflow = 1'b0;
            inexact = 1'b0;
            if (a_nan || b_nan || c_nan) begin
                if (a_snan) result = a | 16'h0200;
                else if (b_snan) result = b | 16'h0200;
                else if (c_snan) result = c | 16'h0200;
                else if (a_nan) result = a | 16'h0200;
                else if (b_nan) result = b | 16'h0200;
                else result = c | 16'h0200;
            end else if (invalid_product) result = 16'h7e00;
            else if (infinite_product && c_inf && (product_sign != c[15])) begin
                result = 16'h7e00;
                invalid = 1'b1;
            end else if (infinite_product) result = {product_sign, 15'h7c00};
            else result = {c[15], 15'h7c00};
        end
    end
endmodule
