CXX      := g++
CXXFLAGS := -std=c++20 -Ofast -w -g -MMD -MP -Iinclude -I. -I$(shell brew --prefix)/include
LDFLAGS  := -L$(shell brew --prefix)/lib -lmatplot -lfftw3

SRCDIR   := src
BUILDDIR := build

# Core library (src/*.cc -> libml.a)
LIB_SRCS := $(wildcard $(SRCDIR)/*.cc)
LIB_OBJS := $(patsubst $(SRCDIR)/%.cc,$(BUILDDIR)/%.o,$(LIB_SRCS))

# Application sources (root directory)
APP_SRCS := main.cc train_loop.cc fourier_train.cc
APP_OBJS := $(patsubst %.cc,$(BUILDDIR)/%.o,$(APP_SRCS))

ALL_OBJS := $(LIB_OBJS) $(APP_OBJS)
DEPS     := $(ALL_OBJS:.o=.d)

TARGET   := program
LIBRARY  := $(BUILDDIR)/libml.a

.PHONY: all lib clean

all: $(TARGET)

lib: $(LIBRARY)

$(LIBRARY): $(LIB_OBJS) | $(BUILDDIR)
	ar rcs $@ $^

$(TARGET): $(APP_OBJS) $(LIBRARY)
	$(CXX) $(CXXFLAGS) $(APP_OBJS) -L$(BUILDDIR) -lml -o $@ $(LDFLAGS)

# Compile library sources
$(BUILDDIR)/%.o: $(SRCDIR)/%.cc | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile app sources from root
$(BUILDDIR)/%.o: %.cc | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -f $(TARGET)
	rm -rf $(BUILDDIR)

-include $(DEPS)
