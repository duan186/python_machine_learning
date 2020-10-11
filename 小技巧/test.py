def f(n):
    if n>=1000:
        return n-3
    else:
        return f(f(n+5))

print(f(84))