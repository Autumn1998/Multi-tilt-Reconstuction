#ifndef MPIBIGDATA_H_
#define MPIBIGDATA_H_
#define  BLOCK_SIZE 1024*1024*121*4
#include <mpi.h>

void MPI_Bcast_BIG(void *p,long long cnt,int MPI_TYPE, int root, MPI_Comm comm);
void MPI_Reduce_BIG(void *send_buffer,void *recv_buffer, long long cnt ,int MPI_TYPE,int REDUCE_TYPE,int root, MPI_Comm comm); 
int MPI_Gatherv_BIG(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,int *recvcounts,long long *displs, MPI_Datatype recvtype,int root,MPI_Comm comm,int cnt,int myid);
int MPI_Scatterv_BIG(void *sendbuf,int *sendcounts,long long *displs, MPI_Datatype sendtype,void *recvbuf,int recvcount,MPI_Datatype recvtype, int root, MPI_Comm comm,int cnt,int myid);

#endif /* MRCFILEIO_H_ */
