VERILATOR ?= verilator
YOSYS ?= yosys
STA ?= sta
export VERILATOR YOSYS STA
.PHONY: verify analysis
verify:
	$(MAKE) -f Makefile test waveform
	$(MAKE) -f mul-fma.mk test waveform
	node analysis/verify-variant.mjs
	node analysis/verify-units.mjs
analysis: verify
	node analysis/run.mjs
