HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin

mat_mult: $(BIN_DIR)/mat_mult

$(BIN_DIR)/mat_mult: mat_mult.cpp $(BIN_DIR)
	$(HIPCC) $(CXXFLAGS) mat_mult.cpp -o $(BIN_DIR)/mat_mult

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: mat_mult clean
