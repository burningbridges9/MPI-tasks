// FirstApp.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "mpi.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <random>
using namespace std;

//mpiexec -n 4

//1
void HelloWorld()
{
	int size, rank, recv;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)
	{
		printf("Hello from process %d\n", rank);
		for (int i = 1; i < size; i++)
		{
			MPI_Recv(&recv, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
				MPI_COMM_WORLD, &st);
			printf("Hello from process %d\n", recv);
		}
	}
	else
	{
		MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
}

//2
void MaxOfVector()
{
	double x[100], globalMax, locMax = DBL_MIN;
	int ProcRank, ProcNum, N = 100;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	// подготовка данных
	if (ProcRank == 0)
	{
		printf("Created array:\n");
		for (int i = 0; i < 100; i++)
		{
			x[i] = rand() % 100 ;
			printf("%.0f ", x[i]);
			if ((i%25==0) && (i!=0))
			{
				printf("\n");
			}
		}
	}

	// рассылка данных на все процессы
	MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// вычисление частичной суммы на каждом из процессов
	// на каждом процессе ведется поиск loc_max вектора x от i1 до i2
	int chunk = N / ProcNum;
	int i1 = chunk * ProcRank;
	int i2 = chunk * (ProcRank + 1);
	if (ProcRank == ProcNum - 1) { i2 = N; }
	for (int i = i1; i < i2; i++)
	{
		if (locMax < x[i])
		{
			locMax = x[i];
		}
	}
	// сборка locMax на процессе с рангом 0 c редукцией
	//MPI_Reduce(&locMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// без редукции
	// сборка locMAx на процессе с рангом 0
	if (ProcRank == 0) {
		globalMax = locMax;
		for (int i = 1; i < ProcNum; i++) {
			MPI_Recv(&locMax, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD,
				&st);
			if (locMax > globalMax)
			{
				globalMax = locMax;
			}
		}
	}
	else // все процессы отсылают свои частичные суммы
		MPI_Send(&locMax, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);


	// вывод результата
	if (ProcRank == 0)
		printf("\nGlobal Max = %.0f", globalMax);
	MPI_Finalize();

}
// 3
void MonteCarloMethod(int n)
{
	int ProcNum, ProcRank, localDotCount = 0, globalDotCount = 0;   //число точек, попавших в круг
	const int N = n;// 1000000;  //общее число точек
	double x, y;    //координаты
	double pi;   //результат

	

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	int chunk = N / ProcNum;
	int i1 = chunk * ProcRank;
	int i2 = chunk * (ProcRank + 1);
	if (ProcRank == ProcNum - 1)
		i2 = N;

	for (int i = i1; i < i2; ++i)
	{
		x = (double)(rand()) / RAND_MAX;
		y = (double)(rand()) / RAND_MAX;

		if (x * x + y * y <= 1)
		{
			++localDotCount;
		}
	}
	MPI_Reduce(&localDotCount, &globalDotCount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if (ProcRank==0)
	{
		pi = (double)4 * globalDotCount / N;
		printf("pi=%f", pi);
	}

	MPI_Finalize();
}

//4
void DataInit(double* &globalArr, double* &localArr, int ProcNum, int ProcRank, int &N , int &chunk)
{
	//для определения размера локальноого массива
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	/*int restPiece = N;
	for (int i = 0; i<ProcRank; i++)
		restPiece = restPiece - restPiece / (ProcNum - i);
	chunkNum = restPiece / (ProcNum - ProcRank);
	localArr = new double[chunkNum];*/

	int k = N / ProcNum;
	int i1 = k * ProcRank;
	int i2 = k * (ProcRank + 1);
	if (ProcRank == ProcNum - 1) i2 = N;
	chunk = i2-i1;
	localArr = new double[chunk];

	globalArr = new double[N];
	// подготовка данных
	if (ProcRank == 0)
	{
		printf("Created global array:\n");
		for (int i = 0; i < N; i++)
		{
			globalArr[i] = rand() % 201 - 100;
			printf("%.0f ", globalArr[i]);
		}
	}
}

void DataDistribution(double* &GlobalVector, double* &LocalVector,
	int Size, int ProcNum, int ProcRank, int chunkNum)
{
	int *pSendNum; // the number of elements sent to the process
	int *pSendInd; // the index of the first data element sent to the process
				   // Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];

	int chunk = Size / ProcNum;

	for (int i = 0; i < ProcNum; i++)
	{
		int i1 = chunk * i;
		int i2 = chunk * (i + 1);
		if (i == ProcNum - 1) i2 = Size;
		pSendNum[i] = i2 - i1;
		pSendInd[i] = i1;
		//if (i != 0) pSendInd[i] = i1 + 1;
	}

	// Scatter the rows
	MPI_Scatterv(GlobalVector, pSendNum, pSendInd, MPI_DOUBLE, LocalVector,
		pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
}
void GetLocalAver(double * &localArr, int chunk, double & locCount, double & locSum)
{	
	for (int i = 0; i < chunk; i++)
	{
		if (localArr[i] > 0)
		{
			locCount++;
			locSum += localArr[i];
		}
	}
}
void Average()
{
	double globalCount, locCount = 0;
	double globalSum, locSum = 0;
	double average = 0;
	int ProcRank, ProcNum, chunkNum, N = 100;
	double * globalArr;
	double * localArr;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	
	//распределение данных по процессам
	DataInit(globalArr, localArr, ProcNum, ProcRank, N, chunkNum);
	DataDistribution(globalArr, localArr, N, ProcNum, ProcRank, chunkNum);
	GetLocalAver(localArr, chunkNum, locCount, locSum);
	

	//сборка locSum на процессе с рангом 0 c редукцией
	MPI_Reduce(&locCount, &globalCount, 1, MPI_DOUBLE,
		MPI_SUM, 0, MPI_COMM_WORLD);
		// без редукции
		// сборка locMAx на процессе с рангом 0
	if (ProcRank == 0) {
		globalSum = locSum;
		//globalCount = locCount;
		for (int i = 1; i < ProcNum; i++) 
		{
			//MPI_Recv(&locCount, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &st);
			//globalCount += locCount;
			MPI_Recv(&locSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &st);
			globalSum += locSum;
		}
		average = globalSum / globalCount;
	}
	else // все процессы отсылают свои частичные суммы
	{
		MPI_Send(&locSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		//MPI_Send(&locCount, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	}


	delete globalArr;
	delete localArr;
	// вывод результата
	if (ProcRank == 0)
		printf("\nGlobalAver = %f", average);

	MPI_Finalize();

}

// 5
void DataInit(double* &globalArrX, double* &globalArrY, double* &localArrX, double* &localArrY, 
	int ProcNum, int ProcRank, int &N, int &chunk)
{
	//для определения размера локальноого массива
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int k = N / ProcNum;
	int i1 = k * ProcRank;
	int i2 = k * (ProcRank + 1);
	if (ProcRank == ProcNum - 1) i2 = N;
	chunk = i2 - i1;
	localArrX = new double[chunk];
	localArrY = new double[chunk];


	globalArrX = new double[N];
	globalArrY = new double[N];
	// подготовка данных
	if (ProcRank == 0)
	{
		printf("Created global arrays:\n");
		for (int i = 0; i < N; i++)
		{
			globalArrX[i] = rand() % 5;
			printf("%.0f ", globalArrX[i]);
		}
		printf("\n");
		for (int i = 0; i < N; i++)
		{
			globalArrY[i] = rand() % 5;
			printf("%.0f ", globalArrY[i]);
		}
		printf("\n");
	}
}

void DataDistribution(double* &GlobalVectorX, double* &GlobalVectorY, double* &LocalVectorX, double* &LocalVectorY,
	int Size, int ProcNum, int ProcRank, int chunkNum)
{
	int *pSendNum; // the number of elements sent to the process
	int *pSendInd; // the index of the first data element sent to the process
				   // Alloc memory for temporary objects
	pSendInd = new int[ProcNum];
	pSendNum = new int[ProcNum];

	// Define the disposition of the vector for current process
	
	int chunk = Size / ProcNum;

	for (int i = 0; i < ProcNum; i++)
	{
		int i1 = chunk * i;
		int i2 = chunk * (i + 1);
		if (i == ProcNum - 1) i2 = Size;
		pSendNum[i] = i2 - i1;
		pSendInd[i] = i1;
		//if (i!=0) pSendInd[i] = i1 + 1;
	}


	// Scatter the rows
	MPI_Scatterv(GlobalVectorX, pSendNum, pSendInd, MPI_DOUBLE,
		LocalVectorX, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(GlobalVectorY, pSendNum, pSendInd, MPI_DOUBLE, 
		LocalVectorY, pSendNum[ProcRank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	// Free the memory
	delete[] pSendNum;
	delete[] pSendInd;
}

void GetLocalSum(double * &localArrX, double * &localArrY, int chunk, double & locSum)
{
	for (int i = 0; i < chunk; i++)
	{
		locSum += localArrX[i] * localArrY[i];
	}
}

void DotProduct()
{
	double globalSum, locSum = 0;
	int ProcRank, ProcNum, chunkNum, N = 5;
	double * globalArrX;
	double * globalArrY;
	double * localArrX;
	double * localArrY;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	//распределение данных по процессам
	DataInit(globalArrX, globalArrY, localArrX, localArrY, ProcNum, ProcRank, N, chunkNum);
	DataDistribution(globalArrX, globalArrY, localArrX, localArrY, N, ProcNum, ProcRank, chunkNum);
	GetLocalSum(localArrX, localArrY, chunkNum, locSum);


	//сборка locSum на процессе с рангом 0 c редукцией
	/*MPI_Reduce(&locSum, &globalSum, 1, MPI_DOUBLE,
	MPI_SUM, 0, MPI_COMM_WORLD);*/

	// без редукции
	// сборка locMAx на процессе с рангом 0
	if (ProcRank == 0) {
		globalSum = locSum;
		for (int i = 1; i < ProcNum; i++)
		{
			MPI_Recv(&locSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD,
				&st);
			globalSum += locSum;
		}
	}
	else // все процессы отсылают свои частичные суммы
		MPI_Send(&locSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

	delete globalArrX, globalArrY;
	delete localArrX, localArrY;
	// вывод результата
	if (ProcRank == 0)
		printf("\nGlobalSum = %0.f", globalSum);

	MPI_Finalize();

}

// 8
void DataInit(double* &globalArr, double* &localArr, double* &resArr ,  int ProcNum, int ProcRank, int &N, int &chunk)
{
	//для определения размера локальноого массива
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int k = N / ProcNum;
	int i1 = k * ProcRank;
	int i2 = k * (ProcRank + 1);
	if (ProcRank == ProcNum - 1) i2 = N;
	chunk = i2 - i1;
	localArr = new double[chunk];
	globalArr = new double[N];
	resArr = new double[N];
	// подготовка данных
	if (ProcRank == 0)
	{
		printf("Created global array:\n");
		for (int i = 0; i < N; i++)
		{
			globalArr[i] = rand() % 10;
			if(i%10 == 0)
				globalArr[i] = 10*(i+10);
			printf("%.0f ", globalArr[i]);
		}
	}
}

void SendRecv()
{
	int ProcRank, ProcNum, chunk;
	double * globalArr;
	double * localArr;
	double * resultArr;
	double * chunkedResArr;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
	int N = ProcNum * 10;
	//распределение данных по процессам
	DataInit(globalArr, localArr, resultArr, ProcNum, ProcRank, N, chunk);
	chunkedResArr = new double[chunk];
	if (ProcRank == 0) {
		//in proc0 copy globarr to localarr
		for (int i = 0; i < chunk; i++)
		{
			localArr[i] = globalArr[i];
		}
		for (int i = 1; i < ProcNum; i++) 
		{
			MPI_Send(globalArr + i*chunk, chunk, MPI_DOUBLE,
				i, 0, MPI_COMM_WORLD);
		}
	}
	else 
	{
		MPI_Recv(localArr, chunk, MPI_DOUBLE, 0,
			0, MPI_COMM_WORLD, &st);
	}
	printf("\nRecieving parts of global array to rank %i:\n", ProcRank);
	for (int j = 0; j < chunk; j++)
	{
		printf("%.0f ", localArr[j]);
	}
	printf("\n");


	// сбор будет на последнем процессе
	if (ProcRank == ProcNum -1) {
		// fill part resArr by localArr in last proc
		for (int i = 0; i < chunk; i++)
		{
			resultArr[ProcRank*chunk + i] = localArr[i];
		}
		for (int i = 0; i < ProcNum - 1; i++)
		{
			
			MPI_Recv(chunkedResArr, chunk, MPI_DOUBLE, i,
				MPI_ANY_TAG, MPI_COMM_WORLD, &st);
			for (int j = 0; j < chunk; j++)
			{
				resultArr[i*chunk + j] = chunkedResArr[j];
			}
		}
	}
	else
	{
		MPI_Send(localArr, chunk, MPI_DOUBLE,
			(ProcNum-1), 0, MPI_COMM_WORLD);
	}

	if (ProcRank == ProcNum - 1)
	{ 
		printf("\nResult array into rank %i:\n", ProcRank);
		for (int j = 0; j < N; j++)
		{
			printf("%.0f ", resultArr[j]);
		}
	}

	MPI_Finalize();

}

// 9
void ReverseLocalArray(double * localArr, int chunk)
{
	reverse(localArr, localArr + chunk);
}
// Function for gathering the result vector
void ResultReplication(double* &GlobalVector, double* &LocalVector,
	int Size, int ProcNum, int ProcRank) {
	int *pReceiveNum; // Number of elements, that current process sends
	int *pReceiveInd; /* Index of the first element from current process
					  in result vector */
	pReceiveNum = new int[ProcNum];
	pReceiveInd = new int[ProcNum];

	int chunk = Size / ProcNum;

	for (int i = 0; i < ProcNum; i++)
	{
		int i1 = chunk * i;
		int i2 = chunk * (i + 1);
		if (i == ProcNum - 1) i2 = Size;
		pReceiveNum[i] = i2 - i1;
		pReceiveInd[i] = i1;
	}
	
	reverse(pReceiveInd, pReceiveInd + ProcNum);
	if (pReceiveNum[ProcNum - 1] != pReceiveNum[0])
	{
		int  k = pReceiveNum[ProcNum - 1] - pReceiveNum[0];
		for (int i = 0; i < ProcNum-1; i++)
		{
			pReceiveInd[i] += k;
		}
	}
	MPI_Gatherv(LocalVector, pReceiveNum[ProcRank], MPI_DOUBLE, 
		GlobalVector, pReceiveNum, pReceiveInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	//Free the memory
	delete[] pReceiveNum;
	delete[] pReceiveInd;
}
void Reverse()
{
	double globalSum, locSum = 0;
	int ProcRank, ProcNum, chunk, incomingChunk, N = 10;
	double * globalArr;
	double * localArr;
	double * globalRevArr;
	double * localRevArr;
	MPI_Status st;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
	MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

	//распределение данных по процессам
	DataInit(globalArr, localArr, globalRevArr,  ProcNum, ProcRank, N, chunk);
	DataDistribution(globalArr, localArr, N, ProcNum, ProcRank, chunk);
	ReverseLocalArray(localArr, chunk);
	ResultReplication(globalArr, localArr, N, ProcNum, ProcRank);

	
	//delete globalArr;
	//delete localArr;
	// вывод результата
	if (ProcRank == 0)
	{
		printf("\nReversed array:\n");
		for (int i = 0; i < N; i++)
		{
			printf("%.0f ", globalArr[i]);
		}
	}
	
	MPI_Finalize();
}



// 10
void GetSendTime(int & size, int & rank)
{
	int  N = 1000000;
	
	double * x = new double[N];
	MPI_Status st;

	// подготовка данных
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			x[i] = rand() % 100;
		}
	}

	double t1, t2, dt;


	int dest = rank + 1;
	int source = rank - 1;
	if (rank == 0)
	{
		source = MPI_PROC_NULL;
	}
	if (rank == size - 1)
	{
		dest = MPI_PROC_NULL;
	}
	t1 = MPI_Wtime();
	MPI_Recv(x, N, MPI_INT,
		source, 1, MPI_COMM_WORLD, &st);
	MPI_Send(x, N, MPI_INT,
		dest, 1, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	dt = t2 - t1;

	// вывод результата
	if (rank == 1)
		printf("\nElapsed time for SEND = %f", dt);
	delete x;
}
void GetSsendTime(int & size, int & rank)
{
	int  N = 1000000;
	double * x = new double[N];
	MPI_Status st;

	// подготовка данных
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			x[i] = rand() % 100;
		}
	}

	double t1, t2, dt;
	

	int dest = rank + 1;
	int source = rank - 1;
	if (rank == 0)
	{
		source = MPI_PROC_NULL;
	}
	if (rank == size - 1)
	{
		dest = MPI_PROC_NULL;
	}
	t1 = MPI_Wtime();
	MPI_Recv(x, N, MPI_INT,
		source, 1, MPI_COMM_WORLD, &st);
	MPI_Ssend(x, N, MPI_INT,
		dest, 1, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	dt = t2 - t1;

	// вывод результата
	if (rank == 1)
		printf("\nElapsed time for SSEND = %f", dt);
	delete x;
}
void GetBsendTime(int & size, int & rank)
{
	int  N = 1000000;
	double * x = new double[N];
	MPI_Status st;

	// подготовка данных
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			x[i] = rand() % 100;
		}
	}

	double t1, t2, dt;

	MPI_Buffer_attach(x,N*sizeof(double) + MPI_BSEND_OVERHEAD);
	int dest = rank + 1;
	int source = rank - 1;
	if (rank == 0)
	{
		source = MPI_PROC_NULL;
	}
	if (rank == size - 1)
	{
		dest = MPI_PROC_NULL;
	}
	t1 = MPI_Wtime();
	MPI_Recv(x, N, MPI_INT,
		source, 1, MPI_COMM_WORLD, &st);
	MPI_Bsend(x, N, MPI_INT,
		dest, 1, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	dt = t2 - t1;

	// вывод результата
	if (rank == 1)
		printf("\nElapsed time for BSEND = %f", dt);
	delete x;
}
void GetRsendTime(int & size, int & rank)
{
	int  N = 1000000;
	double * x = new double[N];
	MPI_Status st;

	// подготовка данных
	if (rank == 0)
	{
		for (int i = 0; i < N; i++)
		{
			x[i] = rand() % 100;
		}
	}

	double t1, t2, dt;


	int dest = rank + 1;
	int source = rank - 1;
	if (rank == 0)
	{
		source = MPI_PROC_NULL;
	}
	if (rank == size - 1)
	{
		dest = MPI_PROC_NULL;
	}
	t1 = MPI_Wtime();
	MPI_Recv(x, N, MPI_INT,
		source, 1, MPI_COMM_WORLD, &st);
	MPI_Rsend(x, N, MPI_INT,
		dest, 1, MPI_COMM_WORLD);
	t2 = MPI_Wtime();
	dt = t2 - t1;

	// вывод результата
	if (rank == 1)
		printf("\nElapsed time for RSEND = %f", dt);
	delete x;
	
}
void GetTime()
{
	int size, rank;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	//GetSendTime(size, rank);
	//GetSsendTime(size, rank);
	GetBsendTime(size, rank);
	//GetRsendTime(size, rank);
	MPI_Finalize();
}



//11
void MessagingBetweenProc()
{
	int size, rank;
	MPI_Status st;
	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	//передача чисел от 0 до size - 1
	int a = 1;
	int dest = rank + 1;
	int source = rank - 1;
	if (rank == 0)
	{
		source = MPI_PROC_NULL;
	}
	if (rank == size - 1)
	{
		dest = MPI_PROC_NULL;
	}
	MPI_Recv(&a, 1, MPI_INT,
		source, 1, MPI_COMM_WORLD, &st);
	a += rank;
	MPI_Send(&a, 1, MPI_INT,
		dest, 1, MPI_COMM_WORLD);
	printf("rank = %i , a = %i\n", rank, a);
	//
	//if (rank == 0)
	//{
	//	source = MPI_PROC_NULL;
	//}
	//else
	//{
	//	if (rank == size - 1)
	//	{
	//		//dest = MPI_PROC_NULL;
	//		MPI_Recv(&a, 1, MPI_INT,
	//			source, 1, MPI_COMM_WORLD, &st);
	//		a += rank;
	//		MPI_Send(&a, 1, MPI_INT,
	//			0, 1, MPI_COMM_WORLD);
	//		printf("rank = %i , a = %i\n", rank, a);

	//	}
	//	else
	//	{
	//		MPI_Recv(&a, 1, MPI_INT,
	//			source, 1, MPI_COMM_WORLD, &st);
	//		a += rank;
	//		MPI_Send(&a, 1, MPI_INT,
	//			dest, 1, MPI_COMM_WORLD);
	//		printf("rank = %i , a = %i\n", rank, a);
	//	}
	//}
	//
	MPI_Finalize();
}


int main(int argc, char* argv[])
{
	// 1
	//HelloWorld();
	// 2
	//MaxOfVector();

	// 3 
	//MonteCarloMethod(10000000);
	//MonteCarloMethod(500000);
	//MonteCarloMethod(1000000);

	// 4 
	//Average();
	
	// 5
	//DotProduct();

	// 8
	//SendRecv();

	// 9 
	Reverse();

	// 10
	//GetTime();

	// 11
	//MessagingBetweenProc();
	//getchar();
    return 0;
}

