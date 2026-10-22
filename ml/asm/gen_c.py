#!/usr/bin/env python3
"""
gen_c.py — builds REAL "C → asm" pairs for the dataset (line A1).

Idea (and why this is more reliable than asking the model to write asm):
  beam writes a C program, and the gcc compiler turns it into asm itself.
  So the asm in the pair is guaranteed correct, a "gold" reference:
  it does not need to be checked with tests; a real compiler produced it.
  The model will later learn exactly this kind of asm (AT&T, x86-64).

What the script does, step by step:
  1. The sample number (0, 1, 2, ...) is deterministically turned into a task
     (task × wording variation; no randomness at all — after a restart
     we continue from exactly the same place).
  2. The prompt goes to the Ollama API (model beam:cloud), and the response is parsed:
     exactly one ```c ... ``` block is required.
  3. The code is compiled TWICE: `gcc -O0 -S` and `gcc -O2 -S` — we get
     two listings of the same program (important: you can see how the optimizer
     changes the code — that is a valuable part of the dataset).
  4. One JSONL line per sample: C source + both listings + flags
     "compiled/failed to compile". Written with flush+fsync (survives a power loss).
  5. The checkpoint state_gen_c.json is updated atomically.

Format A1 — these are "C source → compiler listing" pairs. Tests are
NOT run here: the only quality label is "does it compile". Pairs of format A2
("task → _start program + test run via verifier.py") are deliberately
not mixed in here — that is a separate line B.

Three service items are cut from the listing (for dataset readability):
  - `.file "prog.c"` — the source file name, which we don't need;
  - `.ident "GCC: ..."` — the compiler version;
  - `.section .note...` — empty service sections for the linker.
Everything else (including the .text/.data sections, labels, directives) is kept.
The `-fno-asynchronous-unwind-tables` flag removes the .cfi directive noise at
the compilation stage, and `-fverbose-asm` adds comments with variable names.

Usage:
  python3 gen_c.py --once        one sample (connectivity check)
  python3 gen_c.py --dry-run 5   show 5 prompts without the network
  python3 gen_c.py --status      show the checkpoint
  python3 gen_c.py --self-test   offline check of the gcc part (no network)
  python3 gen_c.py               endless collection (Ctrl+C to stop)
"""

import argparse
import datetime
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import requests

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../ml/asm
RAW_DIR = os.path.join(BASE_DIR, "raw_c")               # shards are written here
STATE_PATH = os.path.join(BASE_DIR, "state_gen_c.json") # checkpoint
LOG_PATH = os.path.join(BASE_DIR, "gen_c.log")          # log

MODEL = "beam:cloud"
HOST = "http://localhost:11434"     # local Ollama daemon (it holds the login)

SYSTEM_PROMPT = (
    "You are an expert C programmer. Reply with exactly one fenced code block "
    "that starts with ```c and contains the complete program. "
    "No explanations, no prose outside the block."
)
TEMP_RANGE = (0.6, 0.95)   # random temperature per request — fewer repeats
JITTER = (0.0, 1.5)        # random addition to the pause (anti-hammering)
REQUEST_TIMEOUT = 900      # seconds for one model response
PAUSE = 2.0                # pause between requests, seconds
MAX_SHARD_BYTES = 8 * 1024 * 1024   # new shard after ~8 MB
MAX_BACKOFF = 3600.0       # maximum wait after a series of network errors
STATE_EVERY_SECONDS = 60   # how often to refresh the checkpoint "just in case"

GCC_TIMEOUT = 30           # seconds per gcc run (in practice: milliseconds)
MAX_EXTRACT_FAILS = 3      # try one prompt this many times, then move on

GCC_FLAGS = [
    "gcc", "-std=c11", "-Wall", "-Wextra", "-pedantic",
    "-fno-asynchronous-unwind-tables",   # no .cfi noise in the listing
    "-fverbose-asm",                     # comments with variable names
]

# ---------------------------------------------------------------------------
# TASK SPACE
# ---------------------------------------------------------------------------
# Each line continues the phrase "Write a complete C program that ...".
# The prompt is deterministic: index -> (task, wording variation).
# Once COMBO is exhausted the prompts repeat, but the answers will differ
# (the temperature is random). Tasks are easy to extend — just add a line.

TASKS = [
    # --- numbers and number theory (70) ---
    "reads one integer n and prints n factorial (n <= 12) using a loop",
    "reads two integers a and b and prints their greatest common divisor via Euclid's algorithm",
    "reads one integer n and prints the n-th Fibonacci number using an iterative loop",
    "reads one integer n and prints 1 if n is prime, otherwise 0",
    "reads one integer n and prints all primes up to n using a sieve of Eratosthenes on a fixed array",
    "reads one integer n and prints the sum of its decimal digits",
    "reads one integer n and prints n with its decimal digits reversed",
    "reads one integer n and prints its binary representation without leading zeros",
    "reads a non-negative integer n and prints the number of set bits computed manually",
    "reads three integers a, b, m and prints (a^b) mod m using fast modular exponentiation",
    "reads one integer n and prints the sum of all its divisors excluding itself",
    "reads one integer n and prints the Collatz sequence length from n down to 1",
    "reads one integer n and prints Euler's totient phi(n) by factorization",
    "reads two integers a and b and prints a^b for small b using a loop",
    "reads one integer n and prints whether it is a perfect square using only integer arithmetic",
    "reads one integer n and prints the largest power of two not exceeding n",
    "reads one integer n and prints the sum 1 + 2 + ... + n using a loop",
    "reads one integer n and prints n! modulo 1000000007 for n <= 100000",
    "reads two integers a and b and prints their least common multiple",
    "reads one integer n and prints the n-th triangular number",
    "reads one integer n and prints the sum 1^2 + 2^2 + ... + n^2",
    "reads one integer n and prints the sum 1^3 + 2^3 + ... + n^3",
    "reads one integer n and prints the sum of floor(n / i) for i from 1 to n",
    "reads one integer n and prints the number of trailing zeros in n factorial",
    "reads one integer n and prints its digital root",
    "reads one integer n and prints the product of its decimal digits",
    "reads one integer n and prints the number of even digits and the number of odd digits",
    "reads one integer n and prints its largest digit and its smallest digit",
    "reads one integer n and prints the sum of its even digits and the sum of its odd digits",
    "reads one integer n and prints 1 if n is an Armstrong number, otherwise 0",
    "reads one integer n and prints 1 if n is a perfect number, otherwise 0",
    "reads one integer n and prints 1 if n is a palindrome in decimal, otherwise 0",
    "reads one integer n and prints the smallest prime greater than n",
    "reads one integer n greater than 2 and prints the largest prime less than n",
    "reads two integers a and b and prints all primes in the inclusive range [a, b]",
    "reads one integer n and prints its prime factorization as p^e terms separated by spaces",
    "reads one integer n and prints the number of distinct prime factors of n",
    "reads one integer n and prints the sum of its prime factors counted with multiplicity",
    "reads one integer n and prints its largest prime factor",
    "reads one integer n and prints 1 if n is a power of two, otherwise 0",
    "reads one positive integer n and prints the exponent of the highest power of two dividing n",
    "reads one integer n and prints 2^n modulo 1000000007",
    "reads one integer n and prints the sum of the first n Fibonacci numbers modulo 1000000007",
    "reads one integer n and prints the n-th Fibonacci number modulo 1000000007 using fast doubling",
    "reads one integer n up to 30 and prints the n-th Catalan number modulo 1000000007",
    "reads one integer n up to 30 and prints C(2n, n) modulo 1000000007",
    "reads two integers n and k and prints C(n, k) modulo 1000000007",
    "reads one integer n and prints the number of odd entries in row n of Pascal's triangle",
    "reads one integer n and prints the number of divisors of n",
    "reads one integer n and prints its aliquot sequence starting at n, one value per line, stopping at 0, at 1, or at a repeated value",
    "reads one integer n and prints the number of steps in the 3n + 1 sequence needed to reach 1",
    "reads two integers a and b and prints gcd(a, b) and lcm(a, b) on one line",
    "reads one integer n and prints the n-th term of the tribonacci sequence with a(1) = a(2) = a(3) = 1",
    "reads one integer n and prints the sum of all integers from 1 to n not divisible by 3 or 5",
    "reads one integer n and prints the number of integers in [1, n] that are not coprime to n",
    "reads one integer n and prints its modular inverse modulo 1000000007 using the extended Euclidean algorithm",
    "reads three integers a, b, m and prints the modular inverse of a modulo m, or -1 if it does not exist",
    "reads one integer n and prints the n-th square number and the n-th triangular number on one line",
    "reads one integer n and prints the sum of the first n odd numbers",
    "reads one integer n and prints the sum of the first n even numbers",
    "reads one integer n and prints the floor of the average of its decimal digits",
    "reads one integer n and prints the number of times the digit 7 appears in it",
    "reads one integer n and prints its digits in words for digits 0 through 9 separated by spaces",
    "reads one integer n and prints the number of ordered representations of n as a sum of two non-negative squares",
    "reads one integer n and prints the integer part of its square root using binary search",
    "reads one integer n and prints the smallest integer k such that k^2 is at least n",
    "reads one integer n and prints the number of perfect squares not exceeding n",
    "reads one integer n greater than 1 and prints its largest proper divisor",
    "reads one integer n and prints the sum of the squares of its decimal digits",
    "reads one positive integer n and prints the number of steps needed to reduce it to 0 by repeatedly subtracting its largest decimal digit",

    # --- strings (60) ---
    "reads one line and prints it reversed",
    "reads one line and prints 1 if it is a palindrome ignoring spaces, otherwise 0",
    "reads one line and prints the number of vowels (a, e, i, o, u, case-insensitive)",
    "reads one line and prints the same line with every lowercase letter shifted by one (a->b, ..., z->a)",
    "reads one line and prints the number of words separated by spaces",
    "reads one line and prints the longest word",
    "reads one line and prints the same line with all spaces removed in place using pointer arithmetic",
    "reads one line and prints its run-length encoding, for example aaabbc becomes a3b2c1",
    "reads one line and prints 1 if all parentheses are balanced, otherwise 0",
    "reads input until EOF and prints the number of lines",
    "reads one line and prints the count of each lowercase letter a-z in 'a:3 b:1' style, skipping zero counts",
    "reads one line and prints its characters sorted in ascending ASCII order without using qsort",
    "reads one line of hexadecimal digits and prints its decimal value",
    "reads one integer in decimal and prints it in lowercase hexadecimal without the 0x prefix",
    "reads one line and converts its words to uppercase without toupper, using ASCII arithmetic",
    "reads one long line and prints 1 if it contains only digits, otherwise 0",
    "reads one line and prints it in lowercase without using tolower",
    "reads one line and prints every second character starting with the first",
    "reads one non-empty line and prints its first character and its last character separated by a space",
    "reads one line and prints the number of consonants",
    "reads one line and prints the number of digits, letters, and spaces as three integers",
    "reads one line and prints its words in reverse order",
    "reads one line and prints the longest word and its length",
    "reads one line and prints the shortest word and its length",
    "reads one line and prints its words sorted by length and then alphabetically",
    "reads one line and prints each duplicate word once in first-occurrence order",
    "reads one line and prints the frequency of each word in first-occurrence order as word:count",
    "reads one line and prints the most frequent word and its count",
    "reads two lines and prints 1 if they are anagrams, otherwise 0",
    "reads two lines and prints 1 if they are equal ignoring case, otherwise 0",
    "reads two lines and prints the length of their longest common prefix",
    "reads two lines and prints the length of their longest common subsequence",
    "reads one line and prints the length of its longest palindromic substring",
    "reads one line and prints its longest palindromic substring, choosing the first one in left-to-right order",
    "reads one line and prints the number of palindromic substrings",
    "reads one line and prints 1 if its brackets (), [], and {} are balanced and properly nested, otherwise 0",
    "reads one line and prints the maximum nesting depth of its parentheses",
    "reads one line and removes duplicate characters, keeping the first occurrence of each",
    "reads one line and prints the first non-repeating character, or - if none exists",
    "reads one line and prints the character with the minimum frequency, choosing the first in case of ties",
    "reads a run-length encoded line of single digits and prints the decoded line, for example a3b2c1 becomes aaabbc",
    "reads one line and prints run-length encoding only for runs longer than one, for example aaabbc becomes a3b2c",
    "reads one line and prints 1 if it is a valid C identifier, otherwise 0",
    "reads one line and prints 1 if it is a decimal integer literal with an optional sign, otherwise 0",
    "reads one line and prints it reversed using recursion",
    "reads one line of at most 8 distinct characters and prints all its permutations in lexicographic order, one per line",
    "reads one line of at most 10 characters and prints all its non-empty subsequences in lexicographic order, one per line",
    "reads one line and prints the number of distinct non-empty subsequences modulo 1000000007",
    "reads one line and prints its lexicographically smallest rotation",
    "reads one line and prints its lexicographically largest rotation",
    "reads two lines and prints 1 if the first is a rotation of the second, otherwise 0",
    "reads one line and prints its simple hash h = h * 31 + c modulo 1000000007, starting from h = 0",
    "reads one line and prints the XOR of all its bytes, excluding the newline",
    "reads one line and prints the number of vowels and the number of consonants on one line",
    "reads one line and prints its words with the first letter of each word capitalized",
    "reads one line and prints its words with the first letter of each word removed",
    "reads one line and prints its words with the letters of each word sorted alphabetically",
    "reads one line and prints its words with duplicate letters removed from each word",
    "reads one line and prints the longest word that appears at least twice, or - if none exists",
    "reads one line and prints the number of occurrences of the substring ab, including overlapping occurrences",

    # --- arrays, sorting, and searching (60) ---
    "reads n then n integers, sorts them ascending with bubble sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with insertion sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with selection sort, and prints them space-separated",
    "reads n then n integers, sorts them ascending with a recursive quicksort, and prints them",
    "reads n then n integers, sorts them ascending with merge sort using a helper array, and prints them",
    "reads n then n integers and prints them using heap sort with an explicit sift-down function",
    "reads n, n sorted integers, and one target x, then prints the index of x via binary search or -1",
    "reads n then n integers and prints the maximum and the second maximum",
    "reads n then n integers and prints their prefix sums space-separated",
    "reads n then n integers and one k, and prints the array rotated right by k positions",
    "reads n then n sorted integers and prints the array with duplicates removed in place",
    "reads n then n integers and prints their median as an integer, using the lower median for even n",
    "reads n then n integers and prints the number of inversions using O(n^2) counting",
    "reads n then n integers and prints the distinct values and their counts",
    "reads n then n integers and prints the longest strictly increasing subsequence length using O(n^2) dynamic programming",
    "reads n then n integers and prints the maximum subarray sum using Kadane's algorithm",
    "reads n, a capacity C, then n item weights, and prints the maximum number of items that fit",
    "reads n then n integers and prints the first index of the minimum and the first index of the maximum",
    "reads n then n integers and prints the difference between the maximum and the minimum",
    "reads n then n integers and prints their average with exactly two digits after the decimal point",
    "reads n then n integers and prints the sum of squared deviations from their floor average",
    "reads n then n integers and prints the mode, choosing the smallest value in case of ties",
    "reads n, k, then n integers and prints the k-th smallest value using 1-based k",
    "reads n, k, then n integers and prints the k-th largest value using 1-based k",
    "reads n then n integers and prints the largest gap between consecutive values after sorting",
    "reads n then n integers and prints the length of the longest contiguous strictly increasing run",
    "reads n, S, then n non-negative integers and prints the length of the longest contiguous subarray with sum at most S",
    "reads n, x, then n integers and prints the number of pairs i < j with a[i] + a[j] = x",
    "reads n, x, then n integers and prints the number of pairs i < j with absolute difference x",
    "reads n then n integers and prints the number of triples i < j < k with a[i] + a[j] + a[k] = 0",
    "reads n then n integers and prints the maximum product of two different elements",
    "reads n then n integers and prints the maximum product of three different elements",
    "reads n then n integers and prints the minimum absolute difference between any two elements",
    "reads n then n integers and prints the length of the longest subarray with equal numbers of even and odd values",
    "reads n then n integers and prints the majority element, or -1 if none exists",
    "reads n then n integers and prints all values occurring more than n/3 times in ascending order, or -1 if none exist",
    "reads n then n integers containing the numbers 1 through n+1 except one, and prints the missing number",
    "reads n then n integers containing exactly one duplicate, and prints the duplicate",
    "reads n then n integers and prints all duplicated values in ascending order, each value once",
    "reads n then n integers and m then m integers and prints their sorted unique intersection, or -1 if it is empty",
    "reads n then n integers and m then m integers and prints their sorted unique union",
    "reads n then n integers and m then m integers and prints their sorted unique symmetric difference, or -1 if it is empty",
    "reads n, x, then n integers and prints the array after removing every occurrence of x",
    "reads n then n integers and prints the array after moving all zeros to the end while preserving order",
    "reads n then n integers and prints the array after moving all negative values to the front while preserving order",
    "reads n, k, then n integers and prints the array rotated left by k positions",
    "reads n then n integers and prints the next greater element for each position, or -1 if none exists",
    "reads n then n integers and prints the previous smaller element for each position, or -1 if none exists",
    "reads n then n non-negative heights and prints the largest rectangle area in the histogram",
    "reads n then n non-negative heights and prints the total trapped rain water",
    "reads n then n integers and prints the maximum non-empty circular subarray sum",
    "reads n, k, then n positive integers and prints the length of the longest subarray with sum exactly k, or 0 if none exists",
    "reads n, k, then n positive integers and prints the number of subarrays with sum exactly k",
    "reads n, k, then n integers and prints the number of subarrays whose sum is divisible by k",
    "reads n, k, then n integers and prints the maximum average of a contiguous subarray of length k with two digits after the decimal point",
    "reads n then n integers and prints the number of pairs i < j with a[i] < a[j]",
    "reads n then n integers and prints the length of the longest strictly decreasing subsequence",
    "reads n then n integers and prints the length of the longest bitonic subsequence",
    "reads n then n integers and prints the number of longest strictly increasing subsequences counted by index choices, modulo 1000000007",
    "reads n then n integers and prints them sorted by absolute value, breaking ties by value ascending",

    # --- matrices and grids (30) ---
    "reads n and m, then an n x m matrix, and prints its transpose",
    "reads two 3 x 3 matrices and prints their product",
    "reads a 10 x 10 grid of '.' and '#' characters and prints the number of connected '#' regions using 4-neighbourhood",
    "reads one integer n and prints an n x n spiral matrix filled with 1 through n^2",
    "reads n and m, then a matrix, and prints its row sums",
    "reads n and m, then a matrix, and prints its column sums",
    "reads n and m, then a matrix, and prints its main diagonal",
    "reads n and m, then a matrix, and prints its anti-diagonal",
    "reads n and m, then a matrix, and prints the sum of its main diagonal",
    "reads an n x n matrix and prints 1 if it is symmetric, otherwise 0",
    "reads an n x n matrix and prints 1 if it is an identity matrix, otherwise 0",
    "reads n, m, a scalar, and a matrix, and prints the matrix multiplied by the scalar",
    "reads two matrices of the same size and prints their element-wise sum",
    "reads an n x n matrix and prints it rotated 90 degrees clockwise",
    "reads an n x n matrix and prints it rotated 90 degrees counterclockwise",
    "reads an n x n matrix and prints it rotated 180 degrees",
    "reads n and m, then a matrix, and prints its elements in spiral order",
    "reads n and m, then a matrix, and prints the sum of every 2 x 2 submatrix in row-major order",
    "reads n and m, then a matrix, and prints a saddle point that is the minimum in its row and the maximum in its column, or -1 if none exists",
    "reads n and m, then a binary grid, and prints the number of islands using 4-neighbourhood",
    "reads n and m, then a binary grid, and prints the size of the largest island",
    "reads n and m, then a binary grid, and prints the shortest path length from (0, 0) to (n-1, m-1) using 4-neighbourhood, or -1 if unreachable",
    "reads n and m, then a grid with 0 for open and 1 for blocked cells, and prints the number of paths from (0, 0) to (n-1, m-1) moving only right and down, modulo 1000000007",
    "reads n and m, then a grid of non-negative costs, and prints the minimum path sum from (0, 0) to (n-1, m-1) moving only right and down",
    "reads n and then n rows of a number triangle, and prints the maximum top-to-bottom path sum moving to adjacent values",
    "reads n and m, then a maze of '.', '#', 'S', and 'E', and prints 1 if there is a path from S to E using 4-neighbourhood, otherwise 0",
    "reads n and m, then a matrix, and prints the sum of all elements on its border",
    "reads n and m, then a matrix, and prints it after setting the entire row and column of every zero in the original matrix to zero",
    "reads n and m, then a matrix, and prints the maximum element of each row and its column index",
    "reads n and m, then a matrix, and prints the number of 2 x 2 submatrices whose four values are equal",

    # --- pointers, structs, memory, and linked structures (40) ---
    "reads n then n integers into a malloc'd array that grows by doubling, prints the sum, and frees everything",
    "defines struct Point { int x; int y; }, reads two points, and prints their squared distance",
    "defines struct Complex { double re; double im; }, reads two complex numbers, and prints their product",
    "builds a singly linked list from n integers read from stdin and prints it reversed using an explicit stack array without recursion",
    "builds a singly linked list from n integers, reverses the list in place by relinking pointers, and prints it",
    "reads n then n integers into a binary search tree and prints them in order using an iterative traversal with an explicit stack",
    "implements a fixed-size int stack on an array of 100 elements driven by commands: p number (push), o (pop and print), q (quit)",
    "implements an int queue with a ring buffer of 100 elements driven by commands: e number (enqueue), d (dequeue and print), q (quit)",
    "reads one line and stores it in a singly linked list of characters, prints it back, and frees every node",
    "defines struct Fraction { int num; int den; }, reads two fractions, and prints their reduced sum",
    "defines struct Fraction { int num; int den; }, reads two fractions, and prints their reduced product",
    "defines struct Date { int day; int month; int year; }, reads two dates, and prints which one is earlier",
    "defines struct Date { int day; int month; int year; }, reads two dates, and prints the absolute number of days between them using Gregorian leap-year rules",
    "defines struct Student { char name[32]; int score; }, reads n students, and prints the name and score of the highest scorer, choosing the first in case of ties",
    "defines struct Student { char name[32]; int score; }, reads n students, and prints them sorted by score descending with ties broken by name ascending",
    "implements a dynamic int array driven by commands: a x (append), g i (print element i), q (quit), freeing the array at the end",
    "implements a singly linked list driven by commands: i x (insert at head), d x (delete the first occurrence), p (print), q (quit)",
    "reads n integers into a doubly linked list and prints it forward and then backward",
    "reads n integers into a circular singly linked list and prints the values twice around starting at the head",
    "reads two sorted linked lists and prints their merged sorted list",
    "reads a sorted linked list and prints it after removing duplicate values",
    "reads a linked list and prints the value of its middle node, using the second middle node for even lengths",
    "reads n, a position p, and n integers, links the tail to node p, and prints 1 if the list has a cycle, otherwise 0",
    "reads n, k, and n integers into a linked list and prints the list after reversing every k consecutive nodes",
    "implements a stack of integers using a linked list driven by commands: p number (push), o (pop and print), t (print top), q (quit)",
    "implements a queue of integers using a linked list driven by commands: e number (enqueue), d (dequeue and print), q (quit)",
    "implements a min-priority queue using a binary heap driven by commands: i x (insert), m (extract minimum and print), q (quit)",
    "implements a hash table with separate chaining for integer keys driven by commands: i x, d x, f x, p, q",
    "implements a hash table with linear probing for integer keys driven by commands: i x, d x, f x, p, q",
    "reads n then n words, stores them in a malloc'd array of char pointers, and prints them one per line",
    "reads n then n words and prints them sorted with qsort and strcmp",
    "reads n then n words into a char pointer array and prints the length of each word",
    "reads n then n words into a linked list, prints the list in reverse, and frees every node",
    "reads n, then for each row its length and values, stores the rows as a jagged array, and prints each row sum",
    "reads a string of 64 zeros and ones, stores it in a uint64_t, and prints the number of set bits",
    "implements a dynamic bitset of n bits driven by commands: s i (set), c i (clear), t i (test and print), p (print all bits), q (quit)",
    "reads n, k, then n integers and prints the last k values in their original order",
    "reads n, k, then n integers and prints the k smallest values in ascending order using a min-heap",
    "reads n, k, then n integers and prints the k largest values in descending order using a max-heap",
    "reads two polynomials as linked lists of coefficients from degree 0 upward and prints their sum coefficients from degree 0 upward",
    "reads n then n words, builds a linked list of strings, prints it, and frees every string and node",

    # --- algorithms, dynamic programming, graphs, and trees (60) ---
    "reads an RPN expression of single-digit numbers and + - * / operators from one line and prints the integer result using an array stack",
    "reads a line of simple additions and subtractions of integers like 3+4-2 and prints the result without using strtol",
    "reads one line of lowercase words and prints them sorted with qsort and a custom comparator",
    "reads a string of Roman numerals and prints its decimal value",
    "reads an integer from 1 to 3999 and prints its Roman numeral representation",
    "reads one integer n and prints the n x n multiplication table with rows space-separated",
    "reads one integer n and prints Pascal's triangle with n rows, each row space-separated",
    "reads one integer n and prints the first n terms of the look-and-say sequence, one per line",
    "reads one line and prints the character with the maximum frequency and its count",
    "reads a command character then n integers, applies the matching function pointer (a=abs, s=square, n=negate), and prints the result",
    "reads n, then n coin denominations, then a target T, and prints the minimum number of coins needed to make T, or -1 if impossible",
    "reads n, then n coin denominations, then a target T, and prints the number of ways to make T modulo 1000000007",
    "reads n, a capacity, then n weights and values, and prints the maximum value of a 0/1 knapsack",
    "reads n, a capacity, then n weights and values, and prints the maximum value of an unbounded knapsack",
    "reads n, a target, then n integers, and prints 1 if some subset sums to the target, otherwise 0",
    "reads n then n integers and prints 1 if they can be partitioned into two equal-sum subsets, otherwise 0",
    "reads n, then n prices for rod lengths 1 through n, and prints the maximum revenue from cutting the rod",
    "reads two lines and prints their edit distance using insertions, deletions, and substitutions",
    "reads two lines and prints the length of their longest common suffix",
    "reads two lines and prints the length of their longest common substring",
    "reads one line and prints the length of its longest palindromic subsequence",
    "reads n, then n+1 matrix dimensions, and prints the minimum number of scalar multiplications for the matrix chain",
    "reads k eggs and n floors and prints the minimum number of trials needed to find the critical floor in the worst case",
    "reads one integer n and prints the number of ways to climb n stairs taking 1, 2, or 3 steps at a time, modulo 1000000007",
    "reads one integer n and prints the number of ways to tile a 2 x n board with dominoes, modulo 1000000007",
    "reads one integer n and prints the number of ways to tile a 3 x n board with dominoes, modulo 1000000007",
    "reads n then n non-negative jump lengths and prints the minimum number of jumps to reach the last index, or -1 if impossible",
    "reads n then n start and end times and prints the maximum number of non-overlapping activities",
    "reads n then n intervals and prints the merged intervals sorted by start time",
    "reads n then n symbol frequencies and prints the total cost of an optimal Huffman code in bits",
    "reads n, m, then m weighted directed edges (u, v, w), a source s, and prints the shortest distances from s using Dijkstra's algorithm",
    "reads n, m, then m weighted directed edges (u, v, w), a source s, and prints the shortest distances from s using Bellman-Ford, or reports a reachable negative cycle",
    "reads n, then an n x n adjacency matrix with -1 for no edge, and prints all-pairs shortest path distances using Floyd-Warshall",
    "reads n, m, then m unweighted edges, a source s, and a target t, and prints the shortest path distance using BFS, or -1 if unreachable",
    "reads n, m, then m unweighted edges and a source s, and prints the DFS traversal order starting at s",
    "reads n, m, then m directed edges and prints a topological order using Kahn's algorithm, or -1 if the graph has a cycle",
    "reads n, m, then m directed edges and prints 1 if the graph contains a directed cycle, otherwise 0",
    "reads n, m, then m undirected edges and prints the number of connected components",
    "reads n, m, then m undirected edges and prints 1 if the graph is bipartite, otherwise 0",
    "reads n, m, then m undirected edges and prints the size of each connected component in ascending order using union-find with path compression",
    "reads n, m, then m weighted undirected edges and prints the total weight of a minimum spanning tree using Kruskal's algorithm, or -1 if the graph is disconnected",
    "reads n, m, then m weighted undirected edges and prints the total weight of a minimum spanning tree using Prim's algorithm, or -1 if the graph is disconnected",
    "reads n then n level-order tree values with -1 for null and prints the tree's in-order traversal",
    "reads n then n level-order tree values with -1 for null and prints the tree height measured in edges",
    "reads n then n level-order tree values with -1 for null and prints the number of leaves",
    "reads n then n level-order tree values with -1 for null and prints the level-order traversal of the mirrored tree, using -1 for null",
    "reads n then n level-order tree values with -1 for null and prints 1 if the tree is a binary search tree, otherwise 0",
    "reads n, then n level-order tree values with -1 for null, then two values, and prints their lowest common ancestor in the binary search tree",
    "reads n then n level-order tree values with -1 for null and prints the level-order traversal",
    "reads n then n level-order tree values with -1 for null and prints the zigzag level-order traversal",
    "reads n then n level-order tree values with -1 for null and prints the tree diameter measured in edges",
    "reads one integer n up to 10 and prints the number of solutions to the n-queens problem",
    "reads one integer n and prints all permutations of 1 through n in lexicographic order, one per line",
    "reads two integers n and k and prints all k-element combinations of 1 through n in lexicographic order, one per line",
    "reads one integer n and prints all valid strings of n pairs of parentheses in lexicographic order, one per line",
    "reads n, m, a grid of letters, and a word, and prints 1 if the word can be traced through adjacent cells without reusing a cell, otherwise 0",
    "reads n and an n x n maze of 0 and 1 and prints the number of paths from (0, 0) to (n-1, n-1) moving only right and down through zeros",
    "reads n, m, a grid of integers, a start row, a start column, and a new colour, and prints the grid after flood fill",
    "reads a string of digits and prints the number of ways to decode it into letters A through Z, modulo 1000000007",
    "reads n then n non-negative integers and prints the maximum sum of non-adjacent elements",
    "reads n then n integers and prints the maximum profit from at most one stock buy and sell transaction",
    "reads n then n prices and prints the maximum profit from unlimited stock buy and sell transactions",
    "reads n then n integers and prints the maximum product of a contiguous subarray",
    "reads n, m, and a grid with obstacles, and prints the number of paths from (0, 0) to (n-1, m-1) moving only right and down, modulo 1000000007",
    "reads n then n balloon values and prints the maximum coins obtained by bursting all balloons, choosing the order optimally",

    # --- bit manipulation (20) ---
    "reads a 32-bit unsigned integer and prints its bits in reverse order",
    "reads a 32-bit unsigned integer and prints the number of set bits",
    "reads a 32-bit unsigned integer and prints the number of trailing zero bits",
    "reads a 32-bit unsigned integer and prints the number of leading zero bits",
    "reads an integer from 0 to 255 and prints 1 if its 8-bit representation has exactly one zero bit, otherwise 0",
    "reads two integers and swaps them using XOR, then prints both values",
    "reads n then n integers where every value appears twice except one, and prints the unique value",
    "reads n then n integers where every value appears twice except two, and prints the two unique values in ascending order",
    "reads n then n integers where every value appears three times except one, and prints the unique value",
    "reads n then n integers and prints the sum of every subset in increasing bitmask order, one per line",
    "reads one integer n and prints the first 2^n Gray code values in decimal, one per line",
    "reads two integers n and width and prints n in binary with exactly width bits, zero-padded",
    "reads a 32-bit unsigned integer and a sequence of commands s k, c k, t k, g k, q and prints the value after each command",
    "reads a 32-bit unsigned integer and k, and prints the value rotated left by k bits",
    "reads a 32-bit unsigned integer and k, and prints the value rotated right by k bits",
    "reads a 32-bit unsigned integer and prints 1 if it has an odd number of set bits, otherwise 0",
    "reads one integer n and prints the smallest power of two greater than or equal to n",
    "reads a 32-bit unsigned integer and prints the index of its highest set bit, or -1 for zero",
    "reads a 32-bit unsigned integer and prints the index of its lowest set bit, or -1 for zero",
    "reads two 32-bit unsigned integers and prints the number of bit positions in which they differ",

    # --- parsing and state machines (20) ---
    "reads a line and parses an integer with an optional leading sign, printing the integer or 0 on failure",
    "reads a line of hexadecimal digits and prints its decimal value",
    "reads a line of binary digits and prints its decimal value",
    "reads a time in HH:MM:SS format and prints the number of seconds since midnight",
    "reads a date in YYYY-MM-DD format and prints its day of the year",
    "reads a line and prints 1 if it is a valid IPv4 address, otherwise 0",
    "reads a valid IPv4 address and prints its 32-bit integer value",
    "reads a line and prints 1 if it is a valid MAC address with colon-separated hexadecimal pairs, otherwise 0",
    "reads a line and prints each identifier, integer, and operator token on its own line",
    "reads a line of arithmetic and prints the number of integer literals, identifiers, and operators as three integers",
    "reads a line with +, -, *, /, parentheses, and integers and prints the integer result with normal precedence and truncating division",
    "reads a sequence of on, off, toggle, and status commands and prints the final state and the number of status queries",
    "reads a sequence of N, S, E, W movement commands with distances and prints the final coordinates starting from (0, 0)",
    "reads n, then n polynomial coefficients from degree 0 upward, and prints the coefficients of the derivative from degree 0 upward",
    "reads a fraction in a/b form and prints it reduced to lowest terms",
    "reads n, then n key=value pairs, and prints the keys in sorted order with their values",
    "reads a line of comma-separated integers and prints their sum and maximum",
    "reads a line of words separated by one or more spaces and prints each word on its own line",
    "reads a line and prints the maximum nesting depth of its curly braces",
    "reads n rows of comma-separated integers with the same number of columns and prints the column sums",

    # --- simulation and games (20) ---
    "reads n, m, and a binary grid, and prints the grid after one step of Conway's Game of Life with dead boundary cells",
    "reads n, m, a binary grid, and a number of steps, and prints the grid after that many steps of Conway's Game of Life with dead boundary cells",
    "reads n, m, a grid of zeros, a start row, a start column, and a number of steps, and prints the grid after that many steps of Langton's ant",
    "reads a 3 x 3 tic-tac-toe board and prints X, O, draw, or invalid",
    "reads a 3 x 3 tic-tac-toe board and prints 1 if it is reachable from a legal game, otherwise 0",
    "reads a square position on an 8 x 8 chessboard and prints the number of legal knight moves from it",
    "reads a square position on an 8 x 8 chessboard and prints the number of squares a queen can move to from it",
    "reads n and a sequence of U, D, L, R commands for a snake of length one on an n x n grid, and prints the final position and whether it hit a wall",
    "reads n, m, a grid with obstacles, a start position, and movement commands, and prints the final position or -1 if a move is blocked",
    "reads the durations of a red, green, and yellow traffic light and a time t, and prints the active colour at time t",
    "reads n floors and a sequence of elevator requests and prints the total distance travelled using first-come-first-served order",
    "reads a sequence of deposit and withdraw commands and prints the final balance, or -1 if any withdrawal would overdraw the account",
    "reads a sequence of add, remove, and query commands for an inventory and prints the final stock and the number of failed removals",
    "reads a price and a list of inserted coins and prints the change using the greedy coin set 25, 10, 5, 1, or -1 if the payment is insufficient",
    "reads a list of blackjack cards and prints the best score not exceeding 21, or bust if every ace assignment exceeds 21",
    "reads five playing cards and prints the rank of the poker hand from 0 for high card through 8 for straight flush",
    "reads two integers n and k and prints the survivor of the Josephus problem using 1-based numbering",
    "reads two integers n and k and prints the elimination order in the hot-potato game using 1-based numbering",
    "reads n rounds of rock-paper-scissors as pairs of characters and prints the first player's score, where a win is 1 and a draw is 0",
    "reads a width and a binary row and prints the next row of elementary cellular automaton rule 110 with zero boundary cells",

    # --- geometry, dates, statistics, hashing, and misc (20) ---
    "reads two points and prints the square of the Euclidean distance between them",
    "reads two points and prints the Manhattan distance between them",
    "reads three points and prints twice the area of the triangle they form",
    "reads three points and prints 1 if they are collinear, otherwise 0",
    "reads three points and prints 1 if they make a clockwise turn, -1 if counterclockwise, and 0 if collinear",
    "reads a rectangle and a point and prints 1 if the point is inside or on the boundary, otherwise 0",
    "reads a circle and a point and prints 1 if the point is inside or on the boundary using squared distances, otherwise 0",
    "reads two line segments and prints 1 if they intersect, otherwise 0",
    "reads n points and prints the number of vertices on their convex hull using the monotone chain algorithm",
    "reads n points and prints the square of the closest pair distance using O(n^2) comparison",
    "reads n polygon vertices in order and prints twice the absolute area using the shoelace formula",
    "reads n polygon vertices in order and a query point and prints 1 if the point is inside the polygon, otherwise 0",
    "reads two axis-aligned rectangles and prints the area of their intersection",
    "reads a year and prints 1 if it is a leap year in the Gregorian calendar, otherwise 0",
    "reads a month and a year and prints the number of days in that month",
    "reads a date and prints the day of the week as an integer from 0 for Monday through 6 for Sunday",
    "reads a date and a number of days and prints the date that many days later",
    "reads n then n integers and prints their floor mean, median, and smallest mode on one line",
    "reads a line of digits and prints how many times each digit 0 through 9 appears, space-separated",
    "reads one line and prints its 32-bit djb2 hash as an unsigned decimal integer",
]

VARIATIONS = [
    "Keep it short and direct, without comments.",
    "Add brief comments explaining the tricky parts.",
    "Use clear variable names and keep functions small.",
    "Handle the empty input case gracefully.",
    "Prefer straightforward loops over clever tricks.",
    "Write it for a beginner audience but fully correct.",
    "Include a few defensive checks on input sizes.",
    "Use fixed-size arrays where possible and avoid VLAs.",
    "Make the code self-contained in one file with a main function.",
    "Write it the way you would in a code review: idiomatic and tidy.",
    "Use size_t for array sizes and loop indices where natural.",
    "Avoid global variables; pass data through function parameters.",
    "Use helper functions for repeated logic instead of one huge main.",
    "Prefer for loops where the iteration count is known.",
    "Prefer while loops where the termination condition is data-driven.",
    "Use switch for command dispatch instead of long if-else chains.",
    "Validate every input conversion and exit cleanly on invalid input.",
    "Read lines with fgets and parse them with sscanf.",
    "Avoid scanf; read the whole line first and then parse it.",
    "Use pointer arithmetic instead of array indexing where it stays readable.",
    "Use const for values that do not change.",
    "Keep every line under 80 columns and use consistent indentation.",
    "Use braces on every block, even single statements.",
    "Avoid goto; structure cleanup with helper functions.",
    "Free every allocation and check for allocation failure.",
    "Use malloc and free for data whose size is known only at runtime.",
    "Prefer fixed-size buffers with explicit bounds checks.",
    "Print exactly one trailing newline and no extra text.",
    "Do not print prompts, menus, or debugging output.",
    "Use only standard C11 headers and libc functions.",
    "Use enum constants for states and command codes.",
    "Keep the output format stable and do not add labels.",
]

COMBO = len(TASKS) * len(VARIATIONS)

DROP_PREFIXES = (".file", ".ident", ".section .note")

FENCE_C = re.compile(r"```[cC]\s*\n(.*?)```", re.S)
FENCE_ANY = re.compile(r"```[^\n]*\n(.*?)```", re.S)


# ---------------------------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------------------------

def prompt_for(index):
    """index -> (task, variation, prompt text). Deterministic."""
    task_i = index % len(TASKS)
    var_i = (index // len(TASKS)) % len(VARIATIONS)
    task, variation = TASKS[task_i], VARIATIONS[var_i]
    prompt = (
        f"Write a complete C program (one file, C11) that {task}. "
        "Read everything from stdin and print the result to stdout. "
        f"{variation}"
    )
    return task, variation, prompt


def extract_c(response):
    """Extracts C code from the response: first look for a ```c block, then any ```.
    If there are several blocks, take the longest one. No blocks — None."""
    blocks = FENCE_C.findall(response)
    if not blocks:
        blocks = FENCE_ANY.findall(response)
    if not blocks:
        return None
    code = max(blocks, key=len).strip()
    return code or None


# ---------------------------------------------------------------------------
# gcc
# ---------------------------------------------------------------------------

def clean_listing(text):
    """Cuts service lines (see the module docstring) and the compiler header.

    The header consists of the leading comment lines `# GNU C11 ...`, `# options passed ...`:
    that is metadata, not asm; we record the gcc version separately in the "gcc" field.
    The `# prog.c:6: ...` comments INSIDE the code are kept — they are a useful
    link from each instruction to a source line."""
    kept = []
    for ln in text.splitlines():
        # compare ignoring indentation: gcc writes "\t.section\t.note..."
        norm = ln.lstrip().replace("\t", " ")
        if norm.startswith(DROP_PREFIXES):
            continue
        kept.append(ln)
    while kept and (not kept[0].strip() or kept[0].startswith("#")):
        kept.pop(0)
    return "\n".join(kept)


def compile_c(c_text, opt):
    """Compiles C text into an asm listing at optimization level opt ('O0'/'O2').

    Returns a dict: ok (bool), asm (str|None), diag (stderr text, truncated).
    The diagnostics are not used for rejection but as data: you can see that the
    model wrote code with warnings — that is also a quality signal.
    """
    with tempfile.TemporaryDirectory(prefix="genc_") as workdir:
        src = os.path.join(workdir, "prog.c")
        out = os.path.join(workdir, "prog.s")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        # pass relative names with cwd=workdir: otherwise gcc writes the
        # absolute path /tmp/genc_XXXX/prog.c into comments — random clutter
        cmd = GCC_FLAGS + [f"-{opt}", "-S", "prog.c", "-o", "prog.s"]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=GCC_TIMEOUT, cwd=workdir)
        except subprocess.TimeoutExpired:
            return {"ok": False, "asm": None,
                    "diag": f"gcc did not finish within {GCC_TIMEOUT} s"}
        except OSError as e:
            return {"ok": False, "asm": None,
                    "diag": f"{type(e).__name__}: {e}"}
        if p.returncode != 0:
            return {"ok": False, "asm": None,
                    "diag": (p.stderr or p.stdout).strip()[:4000]}
        try:
            with open(out, "r", encoding="utf-8") as f:
                listing = clean_listing(f.read())
        except OSError as e:
            return {"ok": False, "asm": None,
                    "diag": f"could not read the listing: {e}"}
        return {"ok": True, "asm": listing, "diag": p.stderr.strip()[:4000]}


# ---------------------------------------------------------------------------
# CHECKPOINT AND FILES
# ---------------------------------------------------------------------------

def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def gcc_version():
    """gcc version — for dataset provenance (the listing header is now removed)."""
    try:
        p = subprocess.run(["gcc", "-dumpfullversion"], capture_output=True,
                           text=True, timeout=5)
        return p.stdout.strip() or "?"
    except OSError:
        return "?"


GCC_VERSION = gcc_version()


def default_state():
    return {
        "version": 1,
        "prompt_index": 0,       # number of the next sample
        "shard_num": 1,          # current file raw_c/shard_NNNNN.jsonl
        "samples_ok": 0,
        "samples_no_c": 0,       # response without a C block (after MAX_EXTRACT_FAILS)
        "samples_bad_c": 0,      # C is present, but gcc compiled neither -O0 nor -O2
        "extract_fails": 0,      # failed attempts for the current index
        "consecutive_errors": 0,
        "min_backoff": None,     # how long the server asked to wait (Retry-After)
        "last_error": None,
        "started_at": None,
        "updated_at": None,
    }


def load_state():
    if not os.path.exists(STATE_PATH):
        st = default_state()
        st["started_at"] = now_iso()
        return st
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        st = json.load(f)
    merged = default_state()
    merged.update(st)
    return merged


def save_state(state):
    """Atomic write: tmp file + os.replace + fsync.
    That way the disk always holds either the old intact version or the new intact one."""
    state["updated_at"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_PATH)
    dir_fd = os.open(os.path.dirname(STATE_PATH), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def shard_path(n):
    return os.path.join(RAW_DIR, f"shard_{n:05d}.jsonl")


def repair_last_line(path):
    """After a crash the last line of a shard may be partially written.
    We truncate the file to the last completed \\n."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if not data or data.endswith(b"\n"):
            return
        cut = data.rfind(b"\n")
        f.truncate(0 if cut < 0 else cut + 1)


def append_sample(shard_num, record):
    """Appends one JSONL line and flushes it to disk. Returns the size."""
    path = shard_path(shard_num)
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return os.path.getsize(path)


def log(msg):
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# NETWORK
# ---------------------------------------------------------------------------

def ask_ollama(prompt, temperature):
    """One request to /api/chat. Returns (response text, eval_count).

    Errors are not swallowed — do_one handles them (backoff/Retry-After)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature},
        "keep_alive": "5m",
    }
    r = requests.post(f"{HOST}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"empty response from the model: keys={list(data.keys())}")
    return content, data.get("eval_count", 0)


# ---------------------------------------------------------------------------
# ONE SAMPLE AND THE LOOP
# ---------------------------------------------------------------------------

def do_one(index, state):
    """Processes one sample.
    Returns a status: "ok" (saved), "skip" (saved as a reject,
    moving on), "net" (network error — we will retry the same index)."""
    task, variation, prompt = prompt_for(index)
    temperature = random.uniform(*TEMP_RANGE)
    log(f"#{index} request (t={temperature:.2f}): {task[:60]}...")
    try:
        response, tokens = ask_ollama(prompt, temperature)
    except requests.HTTPError as e:
        state["consecutive_errors"] += 1
        err = "?"
        if e.response is not None:
            err = f"HTTP {e.response.status_code}"
            ra = e.response.headers.get("retry-after")
            if ra:
                state["min_backoff"] = float(ra)
                err += f", Retry-After={float(ra):.0f}s"
        state["last_error"] = err
        log(f"#{index} ERROR: {err}")
        return "net"
    except Exception as e:
        state["consecutive_errors"] += 1
        state["last_error"] = f"{type(e).__name__}: {e}"
        log(f"#{index} ERROR: {state['last_error']}")
        return "net"

    c_code = extract_c(response)
    if c_code is None:
        state["extract_fails"] += 1
        log(f"#{index} response has no C block "
            f"(attempt {state['extract_fails']}/{MAX_EXTRACT_FAILS})")
        if state["extract_fails"] < MAX_EXTRACT_FAILS:
            return "net"   # retry the same index
        # consistently returns no code — record a reject and move on,
        # otherwise we could get stuck on a "bad" task forever
        record = {
            "id": index, "ts": now_iso(), "model": MODEL, "gcc": GCC_VERSION,
            "task": task, "variation": variation,
            "temperature": round(temperature, 3), "prompt": prompt,
            "extract_ok": False, "raw_response": response[:20000],
            "prompt_tokens": tokens,
            "compile_o0_ok": False, "asm_o0": None, "diag_o0": "no C block",
            "compile_o2_ok": False, "asm_o2": None, "diag_o2": "no C block",
        }
        repair_last_line(shard_path(state["shard_num"]))
        size = append_sample(state["shard_num"], record)
        state["samples_no_c"] += 1
        state["extract_fails"] = 0
        state["prompt_index"] = index + 1
        if size >= MAX_SHARD_BYTES:
            state["shard_num"] += 1
        save_state(state)
        log(f"#{index} reject recorded (no C block), moving on")
        return "skip"

    o0 = compile_c(c_code, "O0")
    o2 = compile_c(c_code, "O2")
    record = {
        "id": index, "ts": now_iso(), "model": MODEL, "gcc": GCC_VERSION,
        "task": task, "variation": variation,
        "temperature": round(temperature, 3), "prompt": prompt,
        "extract_ok": True,
        "c_source": c_code,
        "prompt_tokens": tokens,
        "compile_o0_ok": o0["ok"], "asm_o0": o0["asm"], "diag_o0": o0["diag"],
        "compile_o2_ok": o2["ok"], "asm_o2": o2["asm"], "diag_o2": o2["diag"],
    }
    repair_last_line(shard_path(state["shard_num"]))
    size = append_sample(state["shard_num"], record)
    state["samples_ok"] += 1
    state["extract_fails"] = 0
    state["consecutive_errors"] = 0
    state["last_error"] = None
    state["min_backoff"] = None
    if size >= MAX_SHARD_BYTES:
        state["shard_num"] += 1
    if not (o0["ok"] or o2["ok"]):
        state["samples_bad_c"] += 1
        log(f"#{index} C is present, but gcc failed to compile it: {o0['diag'][:200]}")
    else:
        n0 = len(o0["asm"].splitlines()) if o0["ok"] else 0
        n2 = len(o2["asm"].splitlines()) if o2["ok"] else 0
        log(f"#{index} OK ({tokens} tokens, C lines {len(c_code.splitlines())}, "
            f"asm -O0 {n0} lines, -O2 {n2} lines)")
    state["prompt_index"] = index + 1
    save_state(state)
    return "ok"


def run_forever(state):
    log(f"=== start/resume: continuing from sample #{state['prompt_index']}, "
        f"{COMBO} prompt combinations in total ===")
    backoff = 5.0
    last_state_ts = time.time()
    while True:
        status = do_one(state["prompt_index"], state)
        if status in ("ok", "skip"):
            backoff = 5.0
            wait = PAUSE + random.uniform(*JITTER)
        else:
            # network error: exponential backoff 5s, 10s, 20s ... up to an hour,
            # but not less than Retry-After if the server sent it
            backoff = min(backoff * 2, MAX_BACKOFF)
            wait = max(backoff, state.get("min_backoff") or 0)
            log(f"waiting {wait:.0f}s before retrying")
        if time.time() - last_state_ts > STATE_EVERY_SECONDS:
            save_state(state)
            last_state_ts = time.time()
        time.sleep(wait)


# ---------------------------------------------------------------------------
# SELF-TEST (offline, no network) AND CLI
# ---------------------------------------------------------------------------

SELF_TEST_C = r"""
#include <stdio.h>

static long square(long x) { return x * x; }

int main(void)
{
    int n = 0;
    if (scanf("%d", &n) != 1)
        return 1;
    printf("%ld\n", square(n));
    return 0;
}
"""


def self_test():
    """Checks only the gcc part: compilation, listing and cleaning of service lines.
    No network required. Returns 0 on success."""
    print("1) extracting the C block from the response")
    sample = "Here is the code:\n```c\nint main(void){return 0;}\n```\nDone."
    code = extract_c(sample)
    assert code == "int main(void){return 0;}", repr(code)
    assert extract_c("no block") is None
    print("   ok")

    print("2) gcc -O0/-O2 and listing cleanup")
    o0 = compile_c(SELF_TEST_C, "O0")
    o2 = compile_c(SELF_TEST_C, "O2")
    assert o0["ok"] and o0["asm"], o0["diag"]
    assert o2["ok"] and o2["asm"], o2["diag"]
    for name, res in (("O0", o0), ("O2", o2)):
        asm = res["asm"]
        assert ".file" not in asm, f"{name}: .file still present"
        assert ".ident" not in asm, f"{name}: .ident still present"
        assert ".note" not in asm, f"{name}: .note still present"
        assert "/tmp/" not in asm, f"{name}: temporary path leaked"
        assert not asm.startswith("#"), f"{name}: compiler header still present"
        assert asm.lstrip().startswith(".text"), f"{name}: does not start with .text"
        assert "main:" in asm, f"{name}: no main label"
        assert "square" in asm or "imul" in asm, f"{name}: no square body"
        print(f"   {name}: {len(asm.splitlines())} listing lines — ok")

    print("3) broken C -> compile fail, not an exception")
    bad = compile_c("int main(void) { this is not C }", "O0")
    assert not bad["ok"] and bad["diag"], bad
    print("   ok")

    print("4) prompts are deterministic and cover the space")
    a = prompt_for(0)
    b = prompt_for(0)
    assert a == b
    assert prompt_for(len(TASKS)) != a          # the next variation
    assert prompt_for(COMBO) == a               # the cycle has closed
    print(f"   ok ({len(TASKS)} tasks, {len(VARIATIONS)} variations, "
          f"{COMBO} combinations)")

    print("ALL SELF-TESTS PASSED")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Line A1: C -> gcc listing pairs (O0 and O2) via beam")
    ap.add_argument("--once", action="store_true",
                    help="do exactly one sample and exit (connectivity check)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="show the first N prompts and exit (no network)")
    ap.add_argument("--status", action="store_true", help="show the checkpoint")
    ap.add_argument("--self-test", action="store_true",
                    help="offline check of the gcc part, no network")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    os.makedirs(RAW_DIR, exist_ok=True)
    state = load_state()

    if args.dry_run:
        for i in range(state["prompt_index"],
                       state["prompt_index"] + args.dry_run):
            task, variation, prompt = prompt_for(i)
            print(f"\n#{i}\n{prompt}")
        return

    if args.status:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    if args.once:
        status = do_one(state["prompt_index"], state)
        print(f"status: {status}")
        return

    while True:
        try:
            run_forever(state)
        except KeyboardInterrupt:
            save_state(state)
            log("stopped manually (Ctrl+C)")
            return


if __name__ == "__main__":
    main()
