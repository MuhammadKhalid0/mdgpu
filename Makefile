# Makefile for md-gpu
#
# Loads cuda module on TinyGPU (do `module load cuda/12.6.2` first).
#
# Common targets:
#   make            -> build ./md
#   make release    -> -O3 build (default)
#   make debug      -> debug build
#   make run        -> run a tiny smoke test
#   make tests      -> run all 2-particle sanity tests
#   make scenarios  -> run all visually interesting scenarios
#   make bench      -> performance scan
#   make clean      -> remove build artifacts
#   make distclean  -> also remove generated inputs and outputs

NVCC      ?= nvcc
# sm_75 = Turing (RTX 2080 Ti), sm_86 = Ampere (RTX 3080 / A100=sm_80), sm_70 = V100
# Build a fat binary covering all GPU types present on TinyGPU.
GENCODE   ?= -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86
NVCCFLAGS ?= -std=c++17 -O3 --use_fast_math -Xcompiler "-O3 -Wall" $(GENCODE)
DBGFLAGS  := -std=c++17 -O0 -g -G $(GENCODE)

SRC := src/md.cu
BIN := md

SPH_SRC := src/sph.cu
SPH_BIN := sph

PY  ?= python3

.PHONY: all release debug clean distclean run tests scenarios bench inputs \
        sph_inputs sph_scenarios sph_bench

all: release

release: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(BIN)

debug: $(SRC)
	$(NVCC) $(DBGFLAGS) $(SRC) -o $(BIN)

# ---- SPH mini project ----
$(SPH_BIN): $(SPH_SRC)
	$(NVCC) $(NVCCFLAGS) $(SPH_SRC) -o $(SPH_BIN)

sph_inputs:
	$(PY) scripts/generate_sph_inputs.py


inputs:
	$(PY) scripts/generate_inputs.py

run: $(BIN) inputs
	./$(BIN) -i inputs/two_stable.in -n 200 -d 20 -t 0.002 \
		-o out/run_smoke -p smoke

tests: $(BIN) inputs
	@mkdir -p out
	./$(BIN) -i inputs/two_stable.in    -n 2000 -d 100 -t 0.002 -o out/two_stable    -p p
	./$(BIN) -i inputs/two_attract.in   -n 2000 -d 100 -t 0.002 -o out/two_attract   -p p
	./$(BIN) -i inputs/two_repel.in     -n 2000 -d 100 -t 0.002 -o out/two_repel     -p p

scenarios: $(BIN) inputs
	@mkdir -p out
	./$(BIN) -i inputs/collision.in     -n 4000 -d 50  -t 0.002 -o out/collision     -p p
	./$(BIN) -i inputs/lattice.in       -n 2000 -d 50  -t 0.002 -o out/lattice       -p p
	./$(BIN) -i inputs/cloud.in         -n 3000 -d 50  -t 0.002 -o out/cloud         -p p

# Exercise 3: granular (DEM) scenarios. Spring-dashpot + gravity + box walls.
# k = spring stiffness, -y = damping, -g = gravity magnitude (-z).
dem: $(BIN) inputs
	@mkdir -p out
	./$(BIN) -i inputs/dem_drop.in     -P dem -A naive -L 24 -g 9.81 -k 2000 -y 8 \
		-n 8000  -d 40  -t 0.0008 -o out/dem_drop     -p p
	./$(BIN) -i inputs/dem_pile.in     -P dem -A cells -L 30 -g 9.81 -k 2000 -y 12 \
		-n 12000 -d 100 -t 0.0008 -o out/dem_pile     -p p
	./$(BIN) -i inputs/dem_collapse.in -P dem -A cells -L 40 -g 9.81 -k 2000 -y 12 \
		-n 12000 -d 100 -t 0.0008 -o out/dem_collapse -p p
	./$(BIN) -i inputs/dem_mix.in      -P dem -A cells -L 30 -g 9.81 -k 2000 -y 12 \
		-n 12000 -d 100 -t 0.0008 -o out/dem_mix      -p p

bench: $(BIN) inputs
	$(PY) scripts/bench.py

bench_p2: $(BIN) inputs
	$(PY) scripts/bench_p2.py

# SPH fluid scenarios (dam break, droplet splash, hydrostatic block).
# gravity along -z; k = stiffness, -u = viscosity, -H = smoothing length.
sph_scenarios: $(SPH_BIN) sph_inputs
	@mkdir -p out
	./$(SPH_BIN) -i inputs/sph_dam.in  -A cells -L 32 -H 2.0 -k 15000 -r 1.0 -u 8 -g 9.81 \
		-t 0.002 -n 6000 -d 40 -o out/sph_dam  -p p
	./$(SPH_BIN) -i inputs/sph_drop.in -A cells -L 32 -H 2.0 -k 15000 -r 1.0 -u 8 -g 9.81 \
		-t 0.002 -n 6000 -d 40 -o out/sph_drop -p p
	./$(SPH_BIN) -i inputs/sph_cube.in -A cells -L 32 -H 2.0 -k 15000 -r 1.0 -u 8 -g 9.81 \
		-t 0.002 -n 6000 -d 40 -o out/sph_cube -p p

sph_bench: $(SPH_BIN) sph_inputs
	$(PY) scripts/bench_sph.py

clean:
	rm -f $(BIN) $(SPH_BIN)

distclean: clean
	rm -rf out inputs/*.in
