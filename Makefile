# Project: Neural Network
# Author: Martin Kozlovsky
# Date: 9. 12. 2020

CXX = c++
FLAGS = -std=c++20 -Ofast -fopenmp -Wall -Wextra
NAME = network
SRCS := $(shell find src -name *.cpp)

$(NAME) : $(SRCS)
	$(CXX) $(FLAGS) -o $(NAME) $(SRCS)
