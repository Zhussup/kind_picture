# echo.s — копирует всё из stdin в stdout до EOF.
# Проверяет: read-цикл, пустой ввод (EOF сразу), вывод до 4 КБ за раз.
.section .bss
buf:
    .space 4096

.section .text
.globl _start
_start:
next_read:
    movq $0, %rax          # syscall: read
    movq $0, %rdi          # fd 0 = stdin
    leaq buf(%rip), %rsi
    movq $4096, %rdx
    syscall
    testq %rax, %rax
    jle quit               # 0 = EOF, <0 = ошибка ввода
    movq %rax, %rdx        # выведем ровно столько, сколько прочитали
    movq $1, %rax          # syscall: write
    movq $1, %rdi          # fd 1 = stdout
    leaq buf(%rip), %rsi
    syscall
    jmp next_read
quit:
    movq $60, %rax         # syscall: exit
    xorq %rdi, %rdi
    syscall

.section .note.GNU-stack,"",@progbits