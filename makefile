CPC		= g++
FLAGS	= -O3 -flto -march=native -fopenmp
TARGET	= sdae

$(TARGET):dAE.o DataSet.o LR.o Param.o SdAE.o Utility.o main.cpp
	$(CPC) $(FLAGS) -o $(TARGET) dAE.o DataSet.o LR.o Param.o SdAE.o Utility.o main.cpp
dAE.o:dAE.cpp
	$(CPC) $(FLAGS) -c dAE.cpp 
DataSet.o:DataSet.cpp
	$(CPC) $(FLAGS) -c DataSet.cpp
LR.o:LR.cpp
	$(CPC) $(FLAGS) -c LR.cpp
Param.o:Param.cpp
	$(CPC) $(FLAGS) -c Param.cpp
SdAE.o:SdAE.cpp
	$(CPC) $(FLAGS) -c SdAE.cpp
Utility.o:Utility.cpp
	$(CPC) $(FLAGS) -c Utility.cpp
clean:
	rm -f *.o
	echo Clean done
