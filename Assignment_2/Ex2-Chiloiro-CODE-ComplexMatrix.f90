  module mod_complex16_matrix

  use debugger
  implicit none

!-----------------------------------------------------------------------------
! Description:
!   complex matrix derived type for operating with matrices in double 
!   precision.
!-----------------------------------------------------------------------------

  type complex16_matrix
    ! Elements
    complex*16, dimension(:,:), allocatable :: elem
    ! Dimensions
    integer, dimension(2) :: size
  end type 

  interface randInit
    module procedure :: cdmRandInit
  end interface

  interface zerosInit
  module procedure :: cdmZerosInit
  end interface

  interface operator(.trace.)
    module procedure :: cdmTrace
  end interface

  interface operator(.Adj.)
    module procedure :: cdmAdj
  end interface

  interface to_txt
  module procedure :: cdmToTxt
  end interface


  contains



  subroutine cdmRandInit(size, cmx)
    implicit none
  !-----------------------------------------------------------------------------
  ! Description:
  !   initialize a double precision complex matrix with random complex numbers,
  !   with both real and imaginary part values between 0 and 1.
  !
  ! Pre-conditions: 
  !   size(1) and size(2) must be strictly positive integers.
  !-----------------------------------------------------------------------------

  ! Subroutine arguments
    integer, dimension(2), intent(in) :: size

  ! Return type
    type(complex16_matrix) :: cmx

  ! Local variables
    integer :: ii, jj
    real*8 :: ran_real, ran_im
  !-----------------------------------------------------------------------------
    
  ! check pre-conditions
    if (size(1)<=0 .or. size(2)<=0) then
      call checkpoint(debug=.true., msg='Size must be strictly positive.')
      stop
    end if

  ! executable
    call random_seed()
    ! allocate the correct amount of memory needed
    allocate(cmx%elem(size(1),size(2)))
    ! assign dimensions
    cmx%size = size
    ! assign values
    do ii = 1, size(1)
      do jj = 1, size(2)
        call random_number(ran_real)
        call random_number(ran_im)
        cmx%elem(ii, jj) = cmplx(ran_real, ran_im, kind=8)
      end do
    end do
  end subroutine cdmRandInit



  subroutine cdmZerosInit(size, cmx)
    implicit none
  !-----------------------------------------------------------------------------
  ! Description:
  !   initialize a double precision complex matrix with complex numbers with both 
  !   real and imaginary part values equal 0.
  !
  ! Pre-conditions: 
  !   size(1) and size(2) must be strictly positive integers.
  !-----------------------------------------------------------------------------

  ! Subroutine arguments
    integer, dimension(2), intent(in) :: size

  ! Return type
    type(complex16_matrix) :: cmx
  !-----------------------------------------------------------------------------
    
  ! check pre-conditions
    if (size(1)<=0 .or. size(2)<=0) then
      call checkpoint(debug=.true., msg='Size must be strictly positive.')
      stop
    end if

  ! executable
    ! allocate the correct amount of memory needed
    allocate(cmx%elem(size(1),size(2)))
    ! assign dimensions
    cmx%size = size
    ! assign values
    cmx%elem = cmplx(0., 0., kind=8)
  end subroutine cdmZerosInit



  function cdmTrace(cmx) result(trace)
    implicit none 
  !-----------------------------------------------------------------------------
  ! Description:
  !   Given a complex double matrix cmx, compute its trace.
  !
  ! Pre-conditions:
  !   cmx must be a square matrix.
  !-----------------------------------------------------------------------------

  ! Function arguments
    type(complex16_matrix), intent(in) :: cmx

  ! Return type
    complex*16 :: trace

  ! Local variables
    integer :: ii
  !-----------------------------------------------------------------------------

  ! Check pre-conditions
    if (cmx%size(1) .ne. cmx%size(2)) then
      call checkpoint(debug=.true., msg='Matrix must be square.')
      stop
    end if

  ! Executable
    trace = cmplx(0.,0.)
    do ii = 1, cmx%size(1)
      trace = trace + cmx%elem(ii,ii)
    end do
  end function cdmTrace



  function cdmAdj(cmx) result(adj)
    implicit none 
  !-----------------------------------------------------------------------------
  ! Description:
  !   Given a complex double matrix cmx, it returns its adjoint matrix.
  !
  ! Pre-conditions:
  !   //
  !-----------------------------------------------------------------------------

  ! Function arguments
    type(complex16_matrix), intent(in) :: cmx

  ! Return type
    type(complex16_matrix) :: adj
  !-----------------------------------------------------------------------------

  ! Executable
    ! Allocate
    adj%size(1) = cmx%size(2)
    adj%size(2) = cmx%size(1)
    allocate(adj%elem(adj%size(1),adj%size(2)))
    ! Adjoint
    adj%elem = conjg(transpose(cmx%elem))
  end function cdmAdj



  subroutine cdmToTxt(cmx, filename)
    implicit none
  !-----------------------------------------------------------------------------
  ! Description:
  !   write the given matrix on the file filename.txt. ',' is used as delimiter.
  !
  ! Pre-conditions: 
  !   //
  !-----------------------------------------------------------------------------

  ! Subroutine arguments
    type(complex16_matrix), intent(in) :: cmx
    character(len=*), intent(in) :: filename

  ! Local variables
    integer :: ii, jj
  !-----------------------------------------------------------------------------
    
  ! executable
    open(unit=1, file=filename, status="replace")
    do ii = 1, cmx%size(1)
      do jj = 1 , cmx%size(2)
        ! delimiter only if the element is not the last of the row
        if(jj.ne.cmx%size(2)) then 
          ! correct sign in front of 'i'
          if(aimag(cmx%elem(ii, jj))>=0) then 
            write(1, '(e11.5,a,e11.5,",","  ")', advance='no') real(cmx%elem(ii, jj)), ' + i*',aimag(cmx%elem(ii, jj))
          else 
            write(1, '(e11.5,a,e11.5,",","  ")', advance='no') real(cmx%elem(ii, jj)), ' - i*',abs(aimag(cmx%elem(ii, jj)))
          end if
        ! last element of the row
        else 
          if(aimag(cmx%elem(ii, jj))>=0) then 
            write(1, '(e11.5,a,e11.5)', advance='no') real(cmx%elem(ii, jj)), ' + i*',aimag(cmx%elem(ii, jj))
          else 
            write(1, '(e11.5,a,e11.5)', advance='no') real(cmx%elem(ii, jj)), ' - i*',abs(aimag(cmx%elem(ii, jj)))
          end if
        end if
      end do
      ! new line
      write(1, *) ''
    end do
  end subroutine cdmToTxt

end module mod_complex16_matrix