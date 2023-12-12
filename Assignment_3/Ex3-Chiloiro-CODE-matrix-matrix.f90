function MatrixMultiplication(N, algType) result(elapsed_time)
    !use debugger
    implicit none

!-----------------------------------------------------------------------------
! Description:
!   Matrix-matrix multiplication row-by-row, column-by-column and using MATMUL by 
!   increasing the matrix size from n_min to n_max by step. Save the execution times
!   in three different txt files respectively.
!
! Pre-conditions:
!   n_min, n_max, step must be greater than 0.
!   n_min, step must be less than n_max
!    
! Post-conditions:
!   the output txt files can be read as csv files with comma ',' as delimiter.
!-----------------------------------------------------------------------------

! Function arguments
    ! watch 'Description'
    integer, INTENT(IN) :: N, algType 
    ! output
    real :: elapsed_time

! Local variables
    ! matrices
    real, allocatable :: A(:,:), B(:,:), C(:,:)  
    ! for loops
    integer :: ii, jj, kk
    ! for time measurements
    real :: start_time, end_time 
    
!-----------------------------------------------------------------------------

    ! ! check pre-conditions
    ! if (N<1) then
    !     call checkpoint(debug=.true., msg='N must be greater than 1.')
    !     stop
    ! end if
    ! if ((algType .ne. 1) .and.(algType .ne. 2) .and. (algType .ne. 3)) then
    !     call checkpoint(debug=.true., msg='Invalid algType argument.')
    !     stop
    ! end if


    ! executable
    call random_seed()
    allocate(A(N,N), B(N,N), C(N,N))
    call random_number(A)
    call random_number(B)
    C = 0.

    if (algType == 1) then
        !!!!!!!!!!!! rbc !!!!!!!!!!!!
        ! init time
        call CPU_TIME(start_time)
        ! algorithm
        do ii = 1, N
            do jj = 1, N
                do kk = 1, N
                    C(ii,jj) = C(ii,jj) + A(ii,kk)*B(kk,jj) 
                end do
            end do
        end do
        ! end time
        call CPU_TIME(end_time)
        elapsed_time = end_time - start_time

    else if (algType == 2) then
        !!!!!!!!!!!! cbc !!!!!!!!!!!!
        A = transpose(A)
        call CPU_TIME(start_time)
        do ii = 1, N
            do jj = 1, N
                do kk = 1, N
                    C(ii,jj) = C(ii,jj) + A(kk,ii)*B(kk,jj) 
                end do
            end do
        end do
        call CPU_TIME(end_time)
        elapsed_time = end_time - start_time
    
    else if (algType == 3) then
        !!!!!!!!!!!! MM !!!!!!!!!!!!
        call CPU_TIME(start_time)
        C = matmul(A, B)
        call CPU_TIME(end_time)
        elapsed_time = end_time - start_time
    end if 

    deallocate(A, B, C)


end function MatrixMultiplication
