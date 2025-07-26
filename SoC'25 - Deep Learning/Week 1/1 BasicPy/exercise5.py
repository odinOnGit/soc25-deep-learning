def returnY(x) :
  y = pow(x, 3) - 5*pow(x, 2) + 6*x + 2
  return y

x = int(input("Enter x : "))
print(returnY(x))
