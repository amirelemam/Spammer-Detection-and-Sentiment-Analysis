[Click here to see the English version](README.md)

## Detecção de Spammer e Análise de Sentimentos

### Descrição
2000 tweets foram anotados

### Arquivos

**remove_training**: Retira do arquivo com os dados os tweets usados no treinamento. 
**quebralinhas**: Quebra o arquivo com os dados em 15 arquivos, com 500 mil tweets cada, este número foi definido baseando-se no tamanho médio que o arquivo de 500 mil tweets tinha. Deveria ser um tamanho adequado para o servidor.
**bash**: Gerencia a fila de arquivos, fazendo com que o processamento de cada arquivo seja feito de forma sequencial.
**spammer_detection_and_sentiment_analysis**: Faz a análise de cada tweet para detectar spam e fazer análise de sentimentos. 

### Licença

Estes arquivos estão licenciados sob a [Licença MIT](LICENSE)
