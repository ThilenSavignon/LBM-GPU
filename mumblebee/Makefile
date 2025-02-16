CC = nvcc
CFLAGS = -Werror all-warnings -lineinfo -ccbin g++-13

SRCS = main.cu
OBJS = $(SRCS:.cu=.o)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
	rm -f ./gif/*png