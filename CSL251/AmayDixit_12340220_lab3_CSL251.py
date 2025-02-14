# basic logic gates defined
def AND(a, b):
    return a & b

def OR(a, b):
    return a | b

def NOT(a):
    return ~a

def XOR(a, b):
    return (a & ~b) | (~a & b)

# half adder
def HalfAdder(a, b):
    sum = XOR(a, b)
    carry = AND(a, b)
    return sum, carry

# full adder
def FullAdder(a, b, cin):
    sum1, carry1 = HalfAdder(a, b)
    sum2, carry2 = HalfAdder(sum1, cin)
    cout = OR(carry1, carry2)
    return sum2, cout

def AddSub(A, B, n, subtract=False):
    A_bin = []
    B_bin = []
    
    # convert A to binary
    temp = A
    for i in range(n):
        A_bin.insert(0, temp & 1)
        temp >>= 1
    
    # convert B to binary
    temp = B
    for i in range(n):
        B_bin.insert(0, temp & 1)
        temp >>= 1
    
    # for subtraction, take 2's complement of B
    if subtract:
        # 1's complement
        B_comp = []
        for bit in B_bin:
            B_comp.append(NOT(bit) & 1)
        
        # 2's complement
        carry = 1
        B_bin = []
        for bit in reversed(B_comp):
            sum, carry = FullAdder(bit, 0, carry)
            B_bin.insert(0, sum)
    
    result = []
    carry = 0
    
    # add/sub bit by bit
    for i in range(n-1, -1, -1):
        sum_bit, carry = FullAdder(A_bin[i], B_bin[i], carry)
        result.insert(0, sum_bit)
    
    # convert result back to integer
    result_int = 0
    for bit in result:
        result_int = (result_int << 1) | bit
    
    # check if result is negative and convert to 2's compliment
    if result[0] == 1:
        result_int = result_int | (~((1 << n) - 1))
    
    return result_int

def format_binary(number, n):    
    #just a function to print the binary number
    if number < 0:
        number = number & ((1 << n) - 1)
    return format(number, f'0{n}b')

def BoothsAlgorithm(Multiplicand, Multiplier, n):
    # step1: initializing
    A = 0
    Q_minus_1 = 0
    M = Multiplicand
    Q = Multiplier
    Count = n
    
    while True:
        # get Q0 from Q
        Q0 = Q & 1
        
        # step2: check cases for Q0, Q_minus_1
        if Q0 == 0 and Q_minus_1 == 1:
            A = AddSub(A, M, n)  # A ← A + M
        elif Q0 == 1 and Q_minus_1 == 0:
            A = AddSub(A, M, n, subtract=True)  # A ← A - M
            
        # step3: do arithmetic shift right for A, Q, Q_minus_1
        least_significant_bit_A = A & 1
        most_significant_bit_A = A & (1 << (n-1))
        Q_minus_1 = Q & 1
        
        A = (A >> 1) | most_significant_bit_A
        Q = (Q >> 1) | (least_significant_bit_A << (n-1))
        Count -= 1

        # step4: check if count is 0
        if Count == 0:
            break
            
    return A, Q



n = 4
A = -3    # 0011 in binary
B = -2    # 0010 in binary

def format_2s_complement(num, bits):
    if num < 0:
        num = (1 << bits) + num
    return format(num & ((1 << bits) - 1), f'0{bits}b')

print(f"Adding {A} and {B}:")
print(f"A = {A} (binary: {format_2s_complement(A, n)})")
print(f"B = {B} (binary: {format_2s_complement(B, n)})")

result = AddSub(A, B, n, subtract=False)
print(f"A + B = {result} (binary: {format_2s_complement(result, n)})")

# Test subtraction
print(f"\nSubtracting {B} from {A}:")
result = AddSub(A, B, n, subtract=True)
print(f"A - B = {result} (binary: {format_2s_complement(result, n)})")


n = 3
Multiplicand = -5
Multiplier = 6 

A, Q = BoothsAlgorithm(Multiplicand, Multiplier, n)
print("\nFinal Result:")
print(f"A    = {format_binary(A, n)}")
print(f"Q    = {format_binary(Q, n)}")
print(f"Combined result (A,Q) = {format_binary(A, n)}{format_binary(Q, n)}")