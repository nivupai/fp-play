VERILATOR ?= verilator
export VERILATOR
RTL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD := $(RTL_ROOT)/build
SOURCE := $(RTL_ROOT)/src/fp16_addsub.sv
TOP := fp16_addsub

.PHONY: lint build test waveform clean
lint:
	$(VERILATOR) --lint-only --Wall --top-module $(TOP) $(SOURCE)

build: lint
	mkdir -p $(BUILD)/obj
	$(VERILATOR) --cc --exe --build --trace --Wall --top-module $(TOP) -Mdir $(BUILD)/obj -o fp16_sim -CFLAGS '-std=c++17' $(SOURCE) $(RTL_ROOT)/sim/main.cpp

test: build
	node $(RTL_ROOT)/scripts/verify.mjs

waveform: build
	node $(RTL_ROOT)/scripts/verify.mjs --directed --waveform

clean:
	rm -rf $(BUILD)
