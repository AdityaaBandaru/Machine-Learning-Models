def sumcube(num):
    sum = 0
    i = num
    digit = 0
    while num > 0:
        digit = num % 10
        sum += digit**3
        num//10

    if sum == i:
        return True
    return False

number = int(input("Enter a number: "))

n = sumcube(number)
print(n)