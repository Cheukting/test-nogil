import time

def fibonacci_of(n):
    if n in {0, 1}:  # Base case
        return n
    return fibonacci_of(n - 1) + fibonacci_of(n - 2)  # Recursive case

def run_all():
    total_n = 25
    [fibonacci_of(n) for n in range(total_n)]

if __name__ == '__main__':

    print("==================")
    print("Experiment: fibonacci.py\n")

    total_n = 25

    time_list = []
    for _ in range(50):
        t0 = time.time()
        [fibonacci_of(n) for n in range(total_n)]
        time_list.append(time.time()-t0)
    print(f"Fibonacci of {total_n} takes {sum(time_list)/len(time_list)}s on average")
    print("==================\n")
