#!/bin/bash

#python test_gard_be.py pgd mst 0.01 6
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd constant 0.01 15)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+3+noise constant 0.01 15 & python test_grad_be.py pgd+k+3+noise qp 0.01 15 & python test_grad_be.py pgd+k+3+noise mst 0.01 15)
(trap 'kill 0' SIGINT; python test_grad_be.py gd+k+5+noise constant 0.01 10 0 & python test_grad_be.py gd+k+5+noise constant 0.01 30 1 & python test_grad_be.py gd+k+5+noise constant 0.01 40 2)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 40 0 & python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 40 1 & python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 40 2)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 40 & python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 40 & python test_grad_be_dynamic_k.py pgd+k+3+noise mst 0.01 40)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 50 & python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 50 & python test_grad_be_dynamic_k.py pgd+k+3+noise mst 0.01 50)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 30 & python test_grad_be_dynamic_k.py pgd+k+3+noise  0.01 40 & python test_grad_be_dynamic_k.py pgd+k+3+noise constant 0.01 50)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 30 & python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 40)
# & python test_grad_be_dynamic_k.py pgd+k+3+noise qp 0.01 50)
#(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise mst 0.01 30 & python test_grad_be_dynamic_k.py pgd+k+3+noise mst 0.01 40 & python test_grad_be_dynamic_k.py pgd+k+3+noise mst 0.01 50)

#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+40 constant 0.01 15 & python test_grad_be.py pgd+k+40 qp 0.01 15 & python test_grad_be.py pgd+k+40 mst 0.01 15)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+80 constant 0.01 15 & python test_grad_be.py pgd+k+80 qp 0.01 15 & python test_grad_be.py pgd+k+80 mst 0.01 15)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+160 constant 0.005 20 & python test_grad_be.py pgd+k+160 qp 0.005 20 & python test_grad_be.py pgd+k+160 mst 0.005 20)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+240 constant 0.005 20 & python test_grad_be.py pgd+k+240 qp 0.005 20 & python test_grad_be.py pgd+k+240 mst 0.005 20)
#python test_grad_be.py gd markov 0.01 0
#python test_grad_be.py pgd markov 0.001 0
#python test_grad_be.py gd markov 0.0001 0
#python test_grad_be.py pgd markov 0.0001 0
#(trap 'kill 0' SIGINT; python test_grad.py pgd mst 0.01 4 & python test_grad.py gd mst 0.001 5 & python test_grad.py gd mst 0.0001 6)
#(trap 'kill 0' SIGINT; python test_grad.py pgd mst 0.005 4 & python test_grad.py pgd markov 0.005 5 & python test_grad.py pgd random 0.005 6)
