HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin

vec_add: $(BIN_DIR)/vec_add

$(BIN_DIR)/vec_add: vec_add.cpp $(BIN_DIR)
	$(HIPCC) $(CXXFLAGS) vec_add.cpp -o $(BIN_DIR)/vec_add

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: vec_add clean
