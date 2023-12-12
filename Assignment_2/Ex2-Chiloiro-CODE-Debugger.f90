module debugger

!-----------------------------------------------------------------------------
! Description:
!   This module implements a checkpoint subroutine.
!-----------------------------------------------------------------------------

contains
  
    subroutine checkpoint(debug, msg, var_int, var_real)
    
        implicit none
    
    !-----------------------------------------------------------------------------
    ! Description:
    !     when debug is true, print the string msg and the variables var_int, var_real,
    !     if present. When debug is false, nothing is printed.
    !-----------------------------------------------------------------------------
    
    ! Subroutine arguments
        logical, intent(in) :: debug
        ! optional arguments
        character(len=*),   optional :: msg
        integer,            optional :: var_int
        real,               optional :: var_real

    !-----------------------------------------------------------------------------
    
        if (debug) then 
            if (present(msg)) then 
                print*, msg
            end if
            if (present(var_int)) then 
                print*, var_int
            end if
            if (present(var_real)) then 
                print*, var_real
            end if 
        end if
    
    end subroutine checkpoint

end module debugger