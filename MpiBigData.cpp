#include "MpiBigData.h"


void MPI_Bcast_BIG(void *p,long long cnt,int MPI_TYPE, int root, MPI_Comm comm)
{
	long long off = 0;
	while(BLOCK_SIZE+off<cnt)
	{
	//	printf("BLOCK-SZIE :%d off :%lld\n ",BLOCK_SIZE,off); 
		MPI_Bcast((float *)p+off,BLOCK_SIZE,MPI_TYPE,root,comm);
		off+=BLOCK_SIZE;
	}
	MPI_Bcast((float *)p+off,cnt-off,MPI_TYPE,root,comm);
}

void MPI_Reduce_BIG(void *send_buffer,void *recv_buffer, long long cnt ,int MPI_TYPE,int REDUCE_TYPE,int root, MPI_Comm comm)
{
	long long off = 0;
	while(BLOCK_SIZE+off<cnt)
	{
		MPI_Reduce((float *)send_buffer+off,(float *)recv_buffer+off,BLOCK_SIZE,MPI_TYPE,REDUCE_TYPE,root,comm);
		off+=BLOCK_SIZE;
	}
	MPI_Reduce((float *)send_buffer+off,(float *)recv_buffer+off,cnt-off,MPI_TYPE,REDUCE_TYPE,root,comm);
}

int MPI_Gatherv_BIG(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,int *recvcounts,long long *displs, MPI_Datatype recvtype,int root,MPI_Comm comm,int cnt,int myid)
{
    for(int i=0;i<cnt;i++)
    {
    //   MPI_Send();
    }
    return 1;
}

int MPI_Scatterv_BIG(void *sendbuf,int *sendcounts,long long *displs, MPI_Datatype sendtype,void *recvbuf,int recvcount,MPI_Datatype recvtype, int root, MPI_Comm comm,int cnt,int myid)
{
    return 1;
}

