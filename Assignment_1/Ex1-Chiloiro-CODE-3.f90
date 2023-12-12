program MatrixMultiplication
    implicit none 

    !set the N interval
    integer, parameter :: n_min = 10**2, n_max = 10000, n_step = 200, REP = 1
    ! set file names
    character(100) :: trivial_name = "MM_trivial_fast.txt", column_name = 'MM_column_fast.txt', matmul_name = 'MM_matmul_fast.txt' 

    !variables
    integer :: N !matrix dimension
    real, allocatable :: A(:,:), B(:,:), C(:,:)  !matrices
    integer :: i, j, k, R, l !variables for loops
    real :: start_time, end_time, elapsed_time !times
    
    !for random numbers
    call random_seed()

    !trivial algorithm
    !open a file
    open(unit=1, file=trivial_name, status="replace")
    do N = n_min, n_max, n_step
        allocate(A(N,N), B(N,N), C(N,N))
        call random_number(A)
        call random_number(B)
        C = 0.
        if (N<=1500) then
            R = REP
        else
            R = 1
        end if
        do l = 1, R 
            !measure time 
            call CPU_TIME(start_time)
            !run the algorithm
            do i = 1, N
                do j = 1, N
                    do k = 1, N
                        C(i,j) = C(i,j) + A(i,k)*B(k,j) 
                    end do
                end do
            end do
            call CPU_TIME(end_time)
            print*, 'trivial: done N:', N
            elapsed_time = end_time - start_time
            !save the execution time
            write(1, *) N, ',',elapsed_time
        end do
        deallocate(A, B, C)
    end do
    close(1)

    !column by column algorithm
    !open a file
    open(unit=2, file=column_name, status="replace")
    do N = n_min, n_max, n_step
        allocate(A(N,N), B(N,N), C(N,N))
        call random_number(A)
        call random_number(B)
        !transpose of A 
        A = transpose(A)
        C = 0.
        if (N<=1500) then
            R = REP
        else
            R = 1
        end if
        do l = 1, R 
            !measure time 
            call CPU_TIME(start_time)
            !run the algorithm
            do i = 1, N
                do j = 1, N
                    do k = 1, N
                        C(i,j) = C(i,j) + A(k,i)*B(k,j) 
                    end do
                end do
            end do
            call CPU_TIME(end_time)
            print*, 'column: done N:', N
            elapsed_time = end_time - start_time
            !save the execution time
            write(2, *) N, ',', elapsed_time
        end do
        deallocate(A, B, C)
    end do
    close(2)

    !MATMUL algorithm
    !open a file
    open(unit=3, file=matmul_name, status="replace")
    do N = n_min, n_max, n_step
        allocate(A(N,N), B(N,N), C(N,N))
        call random_number(A)
        call random_number(B)
        C = 0.
        if (N<=1500) then
            R = REP
        else
            R = 1
        end if
        do l = 1, R 
            !measure time 
            call CPU_TIME(start_time)
            !run the algorithm
            C = matmul(A, B)
            call CPU_TIME(end_time)
            print*, 'matmul: done N:', N
            elapsed_time = end_time - start_time
            !save the execution time
            write(3, *) N, ',', elapsed_time
        end do
        deallocate(A, B, C)
    end do
    close(3)

end program MatrixMultiplication