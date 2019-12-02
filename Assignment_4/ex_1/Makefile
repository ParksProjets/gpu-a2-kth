STAGE ?= 1

CXX = pgc++
SRC = $(wildcard src/stage$(STAGE)*.cpp)

OBJ = $(SRC:.cpp=.o)
BIN = ./bin
TARGET = miniWeahter

LDFLAGS = -Wl,-rpath -Wl,/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -Wl,--enable-new-dtags -L/pdc/vol/openmpi/3.0/gcc/6.2.0/lib -lmpi -lpthread -lm
CFLAGS = -I./include -I/pdc/vol/openmpi/3.0/gcc/6.2.0/include -pg -O3
OPENACC_FLAGS = -acc -ta=tesla:cc30,nvidia,lineinfo -Minfo=accel

.PHONY: all dir clean

all: dir $(BIN)/$(TARGET)

dir: $(BIN)
$(BIN):
	mkdir -p $(BIN)


%.o: %.cpp
	$(CXX) $(OPENACC_FLAGS) $(CFLAGS) -c -o $@ $<

$(BIN)/$(TARGET): $(OBJ) lib/libpnetcdf.a | $(BIN)
	$(CXX) $(OPENACC_FLAGS) $(LDFLAGS) -o $@ $^


clean:
	rm -f $(OBJ) $(BIN)/$(TARGET)
