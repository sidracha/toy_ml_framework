CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -g -MMD -MP

SRCS := main.cc tensor_grad.cc tensor.cc linear.cc nn.cc losses.cc
OBJS := $(SRCS:.cc=.o)
DEPS := $(OBJS:.o=.d)

TARGET := program

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS)
