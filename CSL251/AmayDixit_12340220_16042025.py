def mesi_simulation():
    cache_states = ["INVALID", "INVALID", "INVALID", "INVALID"]
    
    while True:
        user_input = input("Enter input triple (a,b,c): ")
        a, b, c = user_input.strip().split(',')
        
        processor = int(a)
        operation = b.strip()
        continue_flag = int(c)
        
        if processor not in [1, 2, 3, 4]:
            print("Invalid processor number. Must be 1, 2, 3, or 4.")
            continue
        if operation not in ['R', 'W']:
            print("Invalid operation. Must be 'R' or 'W'.")
            continue
        if continue_flag not in [0, 1]:
            print("Invalid continue flag. Must be 0 or 1.")
            continue
        
        processor_idx = processor - 1
        
        if operation == 'R':
            if cache_states[processor_idx] == "INVALID":
                if "MODIFIED" in cache_states:
                    modified_idx = cache_states.index("MODIFIED")
                    cache_states[modified_idx] = "SHARED"
                    cache_states[processor_idx] = "SHARED"
                elif "EXCLUSIVE" in cache_states:
                    exclusive_idx = cache_states.index("EXCLUSIVE")
                    cache_states[exclusive_idx] = "SHARED"
                    cache_states[processor_idx] = "SHARED"
                elif "SHARED" in cache_states:
                    cache_states[processor_idx] = "SHARED"
                else:
                    cache_states[processor_idx] = "EXCLUSIVE"
        
        elif operation == 'W':
            for i in range(4):
                if i != processor_idx:
                    cache_states[i] = "INVALID"
            cache_states[processor_idx] = "MODIFIED"
        
        print(f"Cache states after operation: C1: {cache_states[0]}, C2: {cache_states[1]}, C3: {cache_states[2]}, C4: {cache_states[3]}")
        
        if continue_flag == 0:
            break

if __name__ == "__main__":
    print("MESI Cache Coherence Simulation")
    print("All caches start in INVALID state")
    print("Input format: a,b,c where:")
    print("  a = processor number (1-4)")
    print("  b = operation (R for read, W for write)")
    print("  c = continue flag (0 to end, 1 to continue)")
    mesi_simulation()