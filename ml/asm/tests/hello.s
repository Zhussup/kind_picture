# hello.s — печатает "Hello, verifier!\n" и выходит с кодом 0.
# Проверяет: базовый цикл "собрал-слинковал-запустил-stdout".
.section .rodata
msg:
    .ascii "Hello, verifier!\n"
msg_len = . - msg

.section .text
.globl _start
_start:
    movq $1, %rax          # syscall: write
    movq $1, %rdi          # fd 1 = stdout
    leaq msg(%rip), %rsi
    movq $msg_len, %rdx
    syscall
    movq $60, %rax         # syscall: exit
    xorq %rdi, %rdi        # код 0
    syscall

.section .note.GNU-stack,"",@progbits