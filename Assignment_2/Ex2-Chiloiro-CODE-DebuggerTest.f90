program checkpoint_test
    use debugger 
    implicit none 

    ! debug = true
    ! test message 
    call checkpoint(debug=.true., msg='Test')
    ! test int variable
    call checkpoint(debug=.true., var_int=4)
    ! test real variable 
    call checkpoint(debug=.true., var_real=1.5)
    ! test all the three
    call checkpoint(debug=.true., msg='Test all three', var_int=4, var_real=1.5)

    ! debug = false
    call checkpoint(debug=.false., msg='NO', var_int=0, var_real=0.)

end program checkpoint_test