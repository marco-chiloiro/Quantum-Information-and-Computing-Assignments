program test
    use mod_complex16_matrix
    implicit none

    type(complex16_matrix) :: A, B, A_adj, B_adj
    complex*16 :: t_A, t_B

    ! initialize A randomly
    call randInit((/2,2/), A)
    ! initialize B as a matrix of 0s
    call zerosInit((/3,4/), B)
    ! set the diagonal elements of A to 1.+i*2.
    A%elem(1,1) = cmplx(1.d0, 2.d0, kind=8)
    A%elem(2,2) = cmplx(1.d0, 2.d0, kind=8)
    ! set the first element of B to 1.+i*1.
    B%elem(1,1) = cmplx(1.d0, 1.d0, kind=8)
    
    ! trace of A
    t_A = .trace.(A)
    print*, 'Trace of A: ',t_A
    
    ! Adoint of A
    A_adj = .Adj.(A)
    ! Adjoint of B
    B_adj = .Adj.(B)

    ! save A, B, A_adj and B_adj on different txt files
    call to_txt(A, 'A.txt')
    call to_txt(B, 'B.txt')
    call to_txt(A_adj, 'A_adj.txt')
    call to_txt(B_adj, 'B_adj.txt') 

    ! trace of B (not square matrix, then we should expect an error) 
    t_B = .trace.(B)
end program test




