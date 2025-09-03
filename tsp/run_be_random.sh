#!/bin/bash
#(trap 'kill 0' SIGINT; python test_grad_be_old.py pgd+normalize+20 constant 0.01 15 & python test_grad_be_old.py pgd+normalize+20 qp 0.01 15 & python test_grad_be_old.py pgd+normalize+20 mst 0.01 15)
(trap 'kill 0' SIGINT; python test_grad_be_dynamic_k.py pgd+k+3+noise random+constant 0.01 30 & python test_grad_be_dynamic_k.py pgd+k+3+noise random+constant 0.01 40 & python test_grad_be_dynamic_k.py pgd+k+3+noise random+constant 0.01 50)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+3+noise constant 0.01 20 & python test_grad_be.py pgd+k+3+noise qp 0.01 20 & python test_grad_be.py pgd+k+3+noise mst 0.01 20)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+3+noise constant 0.01 50 & python test_grad_be.py pgd+k+3+noise qp 0.01 50 & python test_grad_be.py pgd+k+3+noise mst 0.01 50)

#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+k+80 constant 0.01 50 & python test_grad_be.py pgd+k+80 qp 0.01 50 & python test_grad_be.py pgd+k+80 mst 0.01 50)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+p+0.4 constant 0.01 10 & python test_grad_be.py pgd+p+0.4 qp 0.01 20 & python test_grad_be.py pgd+p+0.4 mst 0.01 20)
#(trap 'kill 0' SIGINT; python test_grad_be.py pgd+p+0.8 constant 0.01 10 & python test_grad_be.py pgd+p+0.8 qp 0.01 20 & python test_grad_be.py pgd+p+0.8 mst 0.01 20)
