import torch
from multiplicar_tensores import multiplicar_tensores

shape_1 = (2, 3)
shape_2 = (3, 4)

resultado = multiplicar_tensores(shape_1, shape_2)

print("Shape resultante:", resultado.shape)
print("Resultado:\n", resultado)