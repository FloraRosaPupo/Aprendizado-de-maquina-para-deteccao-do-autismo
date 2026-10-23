# Experimento Acadêmico de Classificação de Imagens

## Descrição

Experimento de aprendizado de máquina que prepara imagens rotuladas, treina uma rede baseada em VGG19 e calcula métricas de classificação. O nome do repositório reflete o tema do estudo; o código e seus testes não comprovam capacidade de diagnóstico nem utilidade clínica.

## Stack e bibliotecas

Python, TensorFlow/Keras, pandas, NumPy, scikit-learn e matplotlib.

Dependências e versões declaradas: [requirements.txt](requirements.txt).

## Funcionalidades

- Preparação de imagens.
- Treinamento VGG19.
- Relatório de classificação experimental.

## Licença

Não há uma licença de distribuição definida para este repositório. Nenhuma licença MIT, Apache ou equivalente é concedida por este README. As atribuições e os direitos dos materiais de terceiros permanecem preservados.

## Validação e limites

Rótulos, ordenação e rejeição de dataset vazio. A CI não treina o modelo e não comprova utilidade clínica nem validade do dataset.

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

## Organização e manutenção

`dataset.py` concentra a leitura e validação de rótulos; `main.py` contém a execução de treinamento protegida por entry point. A ordem dos arquivos é estável. Uma arquitetura de telas não se aplica a este experimento científico.

Veja [a revisão de desenvolvimento](docs/REVISAO.md) para decisões, testes e limites, e [o guia de contribuição](CONTRIBUTING.md) para o fluxo de manutenção.
