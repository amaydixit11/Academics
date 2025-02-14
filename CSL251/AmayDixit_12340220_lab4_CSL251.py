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

def Division(M, Q, n):
    # initialize
    A = 0
    Q_0 = 0
    Count = n

    while True:
        # step1: left shift (A, Q) together
        A = (A << 1) | (Q >> (n - 1))
        Q = (Q << 1) & ((1 << n) - 1)
        
        # step2: subtract M from A (A = A - M)
        A = AddSub(A, M, n, subtract=True)
        
        # step3: check the sign of A
        if A < 0:
            Q = Q & (~1)  # set Q_0 to 0
            A = AddSub(A, M, n)  # A <- A + M
        else:
            Q = Q | 1  # set Q_0 to 1

        Count -= 1

        if Count == 0:
            break

    return A, Q

n = 4
Divisor = 2
Dividend = 8

A, Q = Division(Divisor, Dividend, n)
print("\nFinal Result of Division:")
print(f"A    = {format_binary(A, n)}")
print(f"Q    = {format_binary(Q, n)}")
print(f"Combined result (A,Q) = {format_binary(A, n)}{format_binary(Q, n)}")