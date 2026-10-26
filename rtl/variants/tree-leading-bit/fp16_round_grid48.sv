// Variant: binary-partition leading-bit search; all finite rounding logic is preserved. magnitude is an exact integer * 2^-48.
module fp16_round_grid48 (
    input logic [80:0] magnitude,
    input logic sign,
    output logic [15:0] result,
    output logic overflow,
    output logic underflow,
    output logic inexact,
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
    int unsigned leading_index;
    logic [127:0] search_vector;
    logic [11:0] rounded_significand;
    always_comb begin
        leading_index = 0;
        normalization_shift = 7'd0;
        discard_count = 7'd24;
        retained_significand = 12'd0;
        guard_bit = 1'b0;
        round_bit = 1'b0;
        sticky_bit = 1'b0;
        increment = 1'b0;
        exponent_rounded = 6'd0;
        rounded_significand = 12'd0;
        result = {sign, 15'd0};
        overflow = 1'b0;
        underflow = 1'b0;
        inexact = 1'b0;
        // Precision-only tininess threshold: (2^-14 - 2^-26) / 2^-48.
        tiny_after_precision_rounding = (magnitude != 81'd0) &&
            (magnitude < (81'd4095 << 22));
        search_vector = {47'd0, magnitude};
        if (|search_vector[127:64]) begin
            leading_index = leading_index + 64;
            search_vector = search_vector >> 64;
        end
        if (|search_vector[127:32]) begin
            leading_index = leading_index + 32;
            search_vector = search_vector >> 32;
        end
        if (|search_vector[127:16]) begin
            leading_index = leading_index + 16;
            search_vector = search_vector >> 16;
        end
        if (|search_vector[127:8]) begin
            leading_index = leading_index + 8;
            search_vector = search_vector >> 8;
        end
        if (|search_vector[127:4]) begin
            leading_index = leading_index + 4;
            search_vector = search_vector >> 4;
        end
        if (|search_vector[127:2]) begin
            leading_index = leading_index + 2;
            search_vector = search_vector >> 2;
        end
        if (|search_vector[127:1]) begin
            leading_index = leading_index + 1;
            search_vector = search_vector >> 1;
        end
        if (magnitude != 81'd0) begin
            normalization_shift = 7'(80 - leading_index);
            if (leading_index >= 34) begin
                exponent_rounded = 6'(leading_index - 33);
                discard_count = 7'(leading_index - 10);
            end
        end
        retained_significand = 12'(magnitude >> discard_count);
        guard_bit = magnitude[discard_count - 7'd1];
        round_bit = magnitude[discard_count - 7'd2];
        for (int unsigned i = 0; i < 81; i = i + 1)
            if (i < (int'(discard_count) - 2)) sticky_bit = sticky_bit | magnitude[i];
        increment = guard_bit && (round_bit || sticky_bit || retained_significand[0]);
        rounded_significand = retained_significand + {11'd0, increment};
        if (rounded_significand[11]) begin
            rounded_significand = rounded_significand >> 1;
            exponent_rounded = exponent_rounded + 6'd1;
        end else if ((exponent_rounded == 6'd0) && rounded_significand[10])
            exponent_rounded = 6'd1;
        overflow = exponent_rounded > 6'd30;
        inexact = guard_bit || round_bit || sticky_bit || overflow;
        underflow = tiny_after_precision_rounding && inexact;
        if (overflow) result = {sign, 15'h7c00};
        else result = {sign, exponent_rounded[4:0], rounded_significand[9:0]};
    end
endmodule
