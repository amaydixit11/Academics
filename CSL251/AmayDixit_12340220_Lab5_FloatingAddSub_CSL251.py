def floating_point_addsub(n, m, S1, S2, E1, E2, subtract=False):
    # Separate sign and magnitude
    class FloatingPointNumber:
        def __init__(self, significand, exponent):
            self.sign = 1 if significand[0] == '1' else 0  # First bit is sign
            self.magnitude = significand[1:]  # Rest is magnitude
            self.exponent = exponent
            
    # Initialize numbers
    X = FloatingPointNumber(S1, E1)
    Y = FloatingPointNumber(S2, E2)
    Z = FloatingPointNumber('0' * (n+1), 0)
    
    # Change sign of Y if subtracting (XOR with 1)
    if subtract:
        Y.sign ^= 1
    
    # Check if X = 0
    if all(bit == '0' for bit in X.magnitude):
        return format_result(Y.sign, Y.magnitude, Y.exponent)
        
    # Check if Y = 0
    if all(bit == '0' for bit in Y.magnitude):
        return format_result(X.sign, X.magnitude, X.exponent)
    
    # Make exponents equal
    while X.exponent != Y.exponent:
        smaller = X if X.exponent > Y.exponent else Y
        smaller.exponent += 1
        smaller.magnitude = '0' + smaller.magnitude[:-1]
        
        if all(bit == '0' for bit in smaller.magnitude):
            other = Y if smaller is X else X
            return format_result(other.sign, other.magnitude, other.exponent)
    
    # Add/subtract magnitudes based on signs
    Z.exponent = X.exponent
    if X.sign == Y.sign:
        Z.sign = X.sign
        Z.magnitude = bin_add(X.magnitude, Y.magnitude)
    else:
        # Determine larger magnitude
        if bin_greater(X.magnitude, Y.magnitude):
            Z.sign = X.sign
            Z.magnitude = bin_sub(X.magnitude, Y.magnitude)
        else:
            Z.sign = Y.sign
            Z.magnitude = bin_sub(Y.magnitude, X.magnitude)
    
    # Check if result is zero
    if all(bit == '0' for bit in Z.magnitude):
        return format_result(0, '0' * n, 0)
    
    # Handle overflow
    if len(Z.magnitude) > n:
        Z.magnitude = '0' + Z.magnitude[:-1]
        Z.exponent += 1
        if is_exponent_overflow(Z.exponent, m):
            return report_overflow()
    
    # Normalize
    while Z.magnitude[0] != '1':
        Z.magnitude = Z.magnitude[1:] + '0'
        Z.exponent -= 1
        if is_exponent_underflow(Z.exponent, m):
            return report_underflow()
    
    # Round result
    Z.magnitude = round_magnitude(Z.magnitude, n-1)
    
    return format_result(Z.sign, Z.magnitude, Z.exponent)

def bin_add(a, b):
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)
    result = ''
    carry = 0
    
    for i in range(max_len-1, -1, -1):
        bit_sum = int(a[i]) + int(b[i]) + carry
        result = str(bit_sum % 2) + result
        carry = bit_sum // 2
    
    if carry:
        result = '1' + result
    return result

def bin_sub(a, b):
    max_len = max(len(a), len(b))
    a = a.zfill(max_len)
    b = b.zfill(max_len)
    result = ''
    borrow = 0
    
    for i in range(max_len-1, -1, -1):
        bit_diff = int(a[i]) - int(b[i]) - borrow
        if bit_diff < 0:
            bit_diff += 2
            borrow = 1
        else:
            borrow = 0
        result = str(bit_diff) + result
    
    return result.lstrip('0') or '0'

def bin_greater(a, b):
    a = a.zfill(max(len(a), len(b)))
    b = b.zfill(max(len(a), len(b)))
    return a > b

def format_result(sign, magnitude, exponent):
    return {
        'significand': str(sign) + magnitude,
        'exponent': exponent
    }

def round_magnitude(mag, n):
    if len(mag) <= n:
        return mag.ljust(n, '0')
    
    rounded = mag[:n]
    if mag[n] == '1':
        # Add 1 to rounded portion
        return bin_add(rounded, '1').zfill(n)
    return rounded

def report_overflow():
    return {'status': 'overflow', 'significand': None, 'exponent': None}

def report_underflow():
    return {'status': 'underflow', 'significand': None, 'exponent': None}

def is_exponent_overflow(exponent, m):
    return exponent >= (1 << (m-1))

def is_exponent_underflow(exponent, m):
    return exponent < -(1 << (m-1))

def main():
    n = 4
    m = 3
    
    S1 = "01001"
    E1 = 1
    S2 = "01011"
    E2 = 1
    
    result_add = floating_point_addsub(n, m, S1, S2, E1, E2, subtract=False)
    print("Addition Result:", result_add)
    
    result_sub = floating_point_addsub(n, m, S1, S2, E1, E2, subtract=True)
    print("Subtraction Result:", result_sub)
    
if __name__ == "__main__":
    main()