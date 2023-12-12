program NumericPrecision
    implicit none
    !integer*2 :: a = 2*10**6,b = 1
    !Error: Arithmetic overflow converting INTEGER(4) to INTEGER(2) at (1). This check can be disabled with the option ‘-fno-range-check’
    integer*4 :: c = 2*10**6,d = 1
    real*4 :: PI_single=4.D0*DATAN(1.D0), square_single = (10.**21.)*(2.**0.5)
    real*8 :: PI_double=4.D0*DATAN(1.D0), square_double = (10.**21.)*(2.**0.5)

    print*, 'Test on integer*4:', c+d 
    print*, 'Test on single precision:', PI_single+square_single
    print*, 'Test on double precision:', PI_double+square_double
    
end program NumericPrecision