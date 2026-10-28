CXXFLAGS ?= -O1
override CXXFLAGS += -std=c++17 -fPIC -I. -Iac_types -MMD -MP

BUILD := build
SRCS := $(wildcard $(BUILD)/*.cpp)
OBJS := $(SRCS:.cpp=.o)
LIB := $(BUILD)/libqtest.so

all: $(LIB)

$(LIB): $(OBJS)
	$(CXX) -shared -o $@ $^

$(BUILD)/%.o: $(BUILD)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD)

-include $(OBJS:.o=.d)

.PHONY: all clean
