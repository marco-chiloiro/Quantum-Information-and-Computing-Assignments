program main
    implicit none
    
    integer, parameter :: n_min = -10, n_max = 1300, step = 300
    call MatrixMultiplication(n_min, n_max, step, .true.)
end program main    




subroutine MatrixMultiplication(n_min, n_max, step, verb)

    use debugger

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

! Subroutine arguments
    ! watch 'Description'
    integer, INTENT(IN) :: n_min, n_max, step
    ! verbosity; if true, print the state of the execution
    logical, INTENT(IN) :: verb

! Local constants
    ! file names
    character(len=10) :: rbr = "rbr.txt", cbc = 'cbc.txt', MM = 'MM.txt' 

! Local variables
    ! matrix dimension
    integer :: N 
    ! matrices
    real, allocatable :: A(:,:), B(:,:), C(:,:)  
    ! for loops
    integer :: ii, jj, kk
    ! for time measurements
    real :: start_time, end_time, elapsed_time 
    
!-----------------------------------------------------------------------------

    ! check pre-conditions
    if (n_min <= 0 .or. n_max <= 0 .or. step <= 0) then
        call checkpoint(debug=.true., msg='Input variables must be positive integer.')
        stop
    end if
    if (n_min >= n_max) then
        call checkpoint(debug=.true., msg='n_max must be larger than n_min.')
        stop
    end if
    if (step >= n_max) then
        call checkpoint(debug=.true., msg='n_max must be larger than step.')
        stop
    end if


    ! executable
    call random_seed()
    ! open txt files
    open(unit=1, file=rbr, status="replace")
    open(unit=2, file=cbc, status="replace")
    open(unit=3, file=MM, status="replace")

    ! matrix size loop
    do N = n_min, n_max, step
        ! init matrices
        allocate(A(N,N), B(N,N), C(N,N))
        call random_number(A)
        call random_number(B)
        C = 0.


        !!!!!!!!!!!! rbr !!!!!!!!!!!!
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
        ! verbosity
        if (verb) then
            print*, 'rbr N:', N
        end if
        ! take the ex. time
        elapsed_time = end_time - start_time
        !save the ex. time
        write(1, *) N, ',',elapsed_time


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
        if (verb) then
            print*, 'cbc N:', N
        end if
        elapsed_time = end_time - start_time
        write(2, *) N, ',', elapsed_time


        !!!!!!!!!!!! MM !!!!!!!!!!!!
        A = transpose(A)
        call CPU_TIME(start_time)
        C = matmul(A, B)
        call CPU_TIME(end_time)
        if (verb) then
            print*, 'MM N:', N
        end if
        elapsed_time = end_time - start_time
        write(3, *) N, ',', elapsed_time


        deallocate(A, B, C)
    end do
    close(1)


end subroutine MatrixMultiplication
