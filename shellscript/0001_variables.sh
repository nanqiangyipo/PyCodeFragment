#!/usr/bin/bash

#no space variable
boy=czm
girl=ylf
act=love
echo $boy $girl $act

# variable with space enclosed with quotes(")
msg2="1. czm love czm"
msg3="2. $boy $act $girl"
echo $msg2 $msg3 
echo ${msg3}asdf
#
echo $((1+2))
