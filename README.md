# Experimento acadêmico de classificação de imagens

Estudo com VGG19/TensorFlow realizado por Flora Rosa, Sabrina Guimarães e Grazielle Stefane. O script treina um classificador usando as classes presentes no conjunto de dados do repositório.

## Executar
Use Python 3.11 em ambiente virtual:
```sh
python -m venv .venv
# Ative o ambiente virtual antes de instalar.
python -m pip install -r requirements.txt
python main.py
```
Execute na raiz do repositório. O script lê `input/AutismDataset/`, baixa pesos ImageNet e salva `vgg19.h5`. Treinamento exige memória e tempo consideráveis; a CI verifica apenas a sintaxe e não treina o modelo.

## Reprodutibilidade e avaliação
A semente é fixada e a divisão de validação é estratificada. A validação não embaralha imagens, para manter previsões alinhadas aos rótulos na matriz de confusão. O treinamento utiliza todos os lotes, incluindo o último incompleto. Hardware e operações não determinísticas ainda podem produzir variações.

## Limitações
Este é um experimento acadêmico, sem validação clínica e sem finalidade diagnóstica. As métricas refletem este conjunto de dados, não demonstram capacidade de diagnosticar pessoas. Origem, autorização de uso, representatividade e possíveis duplicações/vazamentos entre partições devem ser documentados antes de reutilizar o conjunto. A avaliação final independente no conjunto de teste ainda precisa ser implementada.

## Licença
Não há licença de distribuição definida para o código. A licença e os direitos de uso das imagens devem ser verificados separadamente.
