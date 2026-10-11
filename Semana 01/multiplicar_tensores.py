import torch


def multiplicar_tensores(shape_a: tuple, shape_b: tuple):
    if shape_a[-1] != shape_b[-2]:
        raise ValueError("A matriz A deve ter o n de colunas igual o n de linhas da matriz B")

    tensor_a = torch.randn(shape_a)
    tensor_b = torch.randn(shape_b)

    return tensor_a @ tensor_b