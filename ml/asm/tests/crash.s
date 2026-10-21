# crash.s — гарантированный segfault (запись по адресу 0).
# Проверяет: верификатор правильно классифицирует крах как сигнал -11.
.section .text
.globl _start
_start:
    xorq %rax, %rax        # rax = 0
    movb $1, (%rax)        # запись по нулю → SIGSEGV → exit -11

.section .note.GNU-stack,"",@progbits