def padZeroes(M, count): 
    x = "" if count == len(M) else "0"*(count-len(M)) + M
    return x

def twosCompliment(M):
    lst = ["0" if i == "1" else "1" for i in M]
    Q = "0"*(len(lst)-1) + "1" 
    return add("".join(lst), Q)

def add(M, Q): 
    length = max(len(M), len(Q))
    ans = ''
    carry = 0
    for i in range(length - 1, -1, -1):
        sum = carry
        sum += int(M[i]) + int(Q[i])
        ans = str(sum%2) + ans
        carry = 1 if sum == 2 else 0
    return ans

def rightShift(A,q): 
    M = A[0]
    for i in range(1, len(A)):
        M += A[i-1]
    Q = A[-1]
    for j in range(1, len(q)):
        Q += q[j-1]
    Q_minus1 = q[-1]
    return M, Q, Q_minus1

def booth(M, Q):
    M = bin(M_dec).replace("0b", "")
    Q = bin(Q_dec).replace("0b", "")

    minusM = 0
    minusQ = 0

    if (M[0] == "-"):
        M = M.replace("-","")
        minusM = 1
    if (Q[0] == "-"):
        Q = Q.replace("-","")
        minusQ = 1

    count = max(len(M), len(Q)) + 1

    MPositive = padZeroes(M, count) 
    QPositive = padZeroes(Q, count) 
    MNegative = twosCompliment(MPositive) 
    QNegative = twosCompliment(QPositive) 

    M = MNegative if minusM else MPositive
    M_compliment = MPositive if minusM else MNegative
    Q = QNegative if minusQ else QPositive

    A = padZeroes("0", count)
    Q_minus1 = "0" 

    print("\n\nBooth's Algorithm Intermediate Steps: ")
    print("-"*100)
    print(f"count \t A \t Q \t Q_minus1")

    print(f"{str(count)} \t {A} \t {Q} \t {Q_minus1}")

    while (count):
        last_digits = Q[-1] + Q_minus1

        if last_digits == "10":
            A = add(A, M_compliment)
        elif last_digits == "01":
            A = add(A,M)

        A, Q, Q_minus1 = rightShift(A, Q)
        count -= 1

        print(f"{str(count)} \t {A} \t {Q} \t {Q_minus1}")

    answer_binary = A+Q
    print("-"*100)

    if minusM == minusQ:
        ans_decimal = str(int(answer_binary,2))
    else:
        ans_decimal = "-" + str(int(twosCompliment(answer_binary),2))

    return answer_binary, ans_decimal

M_dec = int(input("Enter Multiplicand: "))
Q_dec = int(input("Enter Multiplier: "))

ans_binary, ans_decimal = booth(M_dec, Q_dec)

print("\nbinary answer:  " + ans_binary)
print("decimal answer: " + ans_decimal)