CXX      := g++
CXXFLAGS := -std=c++17 -Ofast -Wall -Wextra -g -MMD -MP -Iinclude -I$(shell brew --prefix)/include
LDFLAGS  := -L$(shell brew --prefix)/lib -lmatplot

SRCDIR   := src
BUILDDIR := build

SRCS := $(wildcard $(SRCDIR)/*.cc)
OBJS := $(patsubst $(SRCDIR)/%.cc,$(BUILDDIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

TARGET := program

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -f $(TARGET)
	rm -rf $(BUILDDIR)

-include $(DEPS)
