VERILATOR ?= verilator
YOSYS ?= yosys
export VERILATOR
RTL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD := $(RTL_ROOT)/build/mul-fma
SOURCES := $(wildcard $(RTL_ROOT)/src/*.sv)
TOP := fp16_mul_fma_compare

.PHONY: lint build test waveform synth clean
lint:
	$(VERILATOR) --lint-only --Wall --top-module $(TOP) $(SOURCES)
build: lint
	mkdir -p $(BUILD)/obj
	$(VERILATOR) --cc --exe --build --trace --Wall --top-module $(TOP) -Mdir $(BUILD)/obj -o fp16_mul_fma_sim -CFLAGS '-std=c++17' $(SOURCES) $(RTL_ROOT)/sim/mul_fma.cpp
test: build
	node $(RTL_ROOT)/scripts/verify-mul-fma.mjs
waveform: build
	node $(RTL_ROOT)/scripts/verify-mul-fma.mjs --directed --waveform
synth:
	mkdir -p $(BUILD)
	cd $(RTL_ROOT) && $(YOSYS) -Q -T -l build/mul-fma/synthesis.log -s scripts/synth-mul-fma.ys
clean:
	rm -rf $(BUILD)
