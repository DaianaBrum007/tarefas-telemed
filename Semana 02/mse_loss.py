import torch

def mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Implementação da Loss Function MSE (Mean Squared Error / Erro Quadrático Médio).
    Recebe 2 tensores (previsão e valor real) e retorna um valor escalar.
    """
    return torch.mean((y_pred - y_true) ** 2)

# Exemplo de uso:
if __name__ == "__main__":
    y_pred = torch.tensor([2.5, 0.0, 2.1], dtype=torch.float32)
    y_true = torch.tensor([3.0, -0.5, 2.0], dtype=torch.float32)

    loss = mse_loss(y_pred, y_true)
    print(f"MSE Loss (Escalar): {loss.item():.4f}")