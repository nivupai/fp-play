// Original FP Play educational binary16 combinational add/subtract datapath.
// Finite values align losslessly on the 2^-24 integer grid (40-bit operands,
// 41-bit sum). This deliberately favors auditability over area optimization.
module fp16_addsub (
    input  logic [15:0] a,
    input  logic [15:0] b,
    input  logic        subtract,
    output logic [15:0] result,
    output logic        invalid,
    output logic        overflow,
    output logic        underflow,
    output logic        inexact,
    // Observation ports: combinational signals, not clocked pipeline stages.
    output logic        effective_b_sign,
    output logic [5:0]  exponent_difference,
    output logic [39:0] aligned_a,
    output logic [39:0] aligned_b,
    output logic [40:0] arithmetic_magnitude,
    output logic [5:0]  normalization_shift,
    output logic [40:0] normalized_magnitude,
    output logic [5:0]  discard_count,
    output logic [11:0] retained_significand,
    output logic        guard_bit,
    output logic        round_bit,
    output logic        sticky_bit,
    output logic        increment,
    output logic [4:0]  exponent_rounded,
    output logic        special_case
);
    logic a_nan, b_nan, a_snan, b_snan, a_inf, b_inf;
    logic [4:0] exponent_a, exponent_b;
    logic [10:0] significand_a, significand_b;
    logic result_sign;
    logic [11:0] rounded_significand;
    int unsigned leading_index;

    always_comb begin
        // 1. Decode/classify. Never apply subtraction to a propagated NaN sign.
        a_nan = (&a[14:10]) && (|a[9:0]);
        b_nan = (&b[14:10]) && (|b[9:0]);
        a_snan = a_nan && !a[9];
        b_snan = b_nan && !b[9];
        a_inf = (&a[14:10]) && !(|a[9:0]);
        b_inf = (&b[14:10]) && !(|b[9:0]);
        effective_b_sign = b[15] ^ subtract;
        exponent_a = (a[14:10] == 5'd0) ? 5'd1 : a[14:10];
        exponent_b = (b[14:10] == 5'd0) ? 5'd1 : b[14:10];
        significand_a = {(a[14:10] != 5'd0), a[9:0]};
        significand_b = {(b[14:10] != 5'd0), b[9:0]};
        exponent_difference = (exponent_a >= exponent_b)
            ? {1'b0, exponent_a} - {1'b0, exponent_b}
            : {1'b0, exponent_b} - {1'b0, exponent_a};

        // Complete defaults prevent inferred latches, including special paths.
        result = 16'h0000;
        invalid = 1'b0;
        overflow = 1'b0;
        // Every finite add/sub result is on the 2^-24 grid. Tiny results are
        // exactly representable, so tiny AND inexact can never hold here.
        underflow = 1'b0;
        inexact = 1'b0;
        aligned_a = 40'd0;
        aligned_b = 40'd0;
        arithmetic_magnitude = 41'd0;
        result_sign = 1'b0;
        leading_index = 0;
        normalization_shift = 6'd0;
        normalized_magnitude = 41'd0;
        discard_count = 6'd0;
        retained_significand = 12'd0;
        guard_bit = 1'b0;
        round_bit = 1'b0;
        sticky_bit = 1'b0;
        increment = 1'b0;
        rounded_significand = 12'd0;
        exponent_rounded = 5'd0;
        special_case = a_nan || b_nan || a_inf || b_inf;

        // 2. Specials: first signaling, otherwise first quiet, original A/B order.
        if (a_nan || b_nan) begin
            invalid = a_snan || b_snan;
            if (a_snan) result = a | 16'h0200;
            else if (b_snan) result = b | 16'h0200;
            else if (a_nan) result = a | 16'h0200;
            else result = b | 16'h0200;
        end else if (a_inf || b_inf) begin
            if (a_inf && b_inf && (a[15] != effective_b_sign)) begin
                result = 16'h7e00;
                invalid = 1'b1;
            end else if (a_inf) result = {a[15], 15'h7c00};
            else result = {effective_b_sign, 15'h7c00};
        end else begin
            // 3. Alignment: effective biased exponent 1 has shift zero. Keep
            // every bit of both operands; no sticky compression before cancellation.
            aligned_a = {29'd0, significand_a} << (exponent_a - 5'd1);
            aligned_b = {29'd0, significand_b} << (exponent_b - 5'd1);
            // 4. Magnitude arithmetic with one extra carry bit.
            if (a[15] == effective_b_sign) begin
                arithmetic_magnitude = {1'b0, aligned_a} + {1'b0, aligned_b};
                result_sign = a[15];
            end else if (aligned_a >= aligned_b) begin
                arithmetic_magnitude = {1'b0, aligned_a} - {1'b0, aligned_b};
                result_sign = (aligned_a == aligned_b) ? 1'b0 : a[15];
            end else begin
                arithmetic_magnitude = {1'b0, aligned_b} - {1'b0, aligned_a};
                result_sign = effective_b_sign;
            end
            // 5. Bounded priority encoder / leading-zero normalization.
            for (int unsigned i = 0; i < 41; i = i + 1) begin
                if (arithmetic_magnitude[i]) leading_index = i;
            end
            if (arithmetic_magnitude != 41'd0) begin
                normalization_shift = 6'(40 - leading_index);
                normalized_magnitude = arithmetic_magnitude << normalization_shift;
                if (leading_index >= 10) exponent_rounded = 5'(leading_index - 9);
                if (leading_index > 10) discard_count = 6'(leading_index - 10);
            end
            retained_significand = 12'(arithmetic_magnitude >> discard_count);
            // 6. G/R/S are generated from the exact result, after cancellation.
            if (discard_count >= 6'd1) guard_bit = arithmetic_magnitude[discard_count - 6'd1];
            if (discard_count >= 6'd2) round_bit = arithmetic_magnitude[discard_count - 6'd2];
            for (int unsigned i = 0; i < 41; i = i + 1) begin
                if ((discard_count > 6'd2) && (i < (int'(discard_count) - 2)))
                    sticky_bit = sticky_bit | arithmetic_magnitude[i];
            end
            // 7. Round-to-nearest, ties-to-even, then carry renormalization.
            increment = guard_bit && (round_bit || sticky_bit || retained_significand[0]);
            rounded_significand = retained_significand + {11'd0, increment};
            if (rounded_significand[11]) begin
                rounded_significand = rounded_significand >> 1;
                exponent_rounded = exponent_rounded + 5'd1;
            end
            inexact = guard_bit || round_bit || sticky_bit;
            // 8. Encode zero/subnormal/normal, or finite-range overflow.
            if (exponent_rounded == 5'd31) begin
                result = {result_sign, 15'h7c00};
                overflow = 1'b1;
                inexact = 1'b1;
            end else result = {result_sign, exponent_rounded, rounded_significand[9:0]};
        end
    end
endmodule
