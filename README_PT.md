[Click here to see the English version](README.md)

## Detecção de Spammer e Análise de Sentimentos

Este projeto consiste na análise de um grupo de 7.189.160 tweets, que foram filtrados a partir de hashtags relacionadas aos ataques terroristas em Paris, em 13 de novembro de 2015. 

O objetivo principal é identificar spammers através das características do usuário, que são obtidas nos metadados do tweet. O objetivo secundário é identificar sentimentos experimentados pelos usuários durante a postagem de seus tweets.

Foi escolhida a análise dos metadados e não do texto do tweet, porque este método é tido na literatura como a segunda melhor maneira de detecção de spam, com eficácia superada apenas pelo estudo de relacionamentos de usuários, através de grafos.

A opção por não fazer o estudo de relacionamentos de usuários deve-se ao seu alto custo computacional e utilização de técnicas avançadas, sendo que este estudo tem por objetivo ser uma análise introdutória.

### Detecção de SPAM

Da base, 1.200 tweets foram coletados de forma aleatória da amostra e anotados manualmente como "SPAM" ou "NÃO SPAM".

Estes tweets foram usados para treinar os três classificadores, criados a partir dos algorítmos Naïve Bayes Bernoulli, Naïve Bayes Multinomial e Decision Tree. 

O quarto classificador foi feito baseando-se nos critérios propostos por [El Azab et al. (2016)](http://waset.org/publications/10003176/fake-account-detection-in-twitter-based-on-minimum-weighted-feature-set), que dá um peso a cada característica do tweet, conforme tabela abaixo:

| Atributo                                         | Peso   |
| :----------------------------------------------: |:------:|
| A conta tem ao menos 30 seguidores               | 0.53   |
| A conta foi geolocalizada                        | 0.85   |
| Foi incluida nos favoritos de outro usuário      | 0.85   |
| Usou hashtag em ao menos um tweet                | 0.85   |
| Fez login no Twitter usando um iPhone            | 0.917  |
| Mencionado por um usuário do Twitter             | 1      |
| Publicou ao menos 50 tweets                      | 0.01   |
| Foi incluido na lista de outro usuário           | 0.45   |
| (2*número de seguidores) _ (numero de seguidos)  | 0.5    |
| Usuário tem ao menos uma lista de favoritos      | 0.17   |

Com os quatro classificadores treinados, o tweet era indicado como SPAM caso a maioria dos classificadores  apontasse como tal. Caso houvesse empate, a classificação feita pelos critérios de [El Azab et al. (2016)](http://waset.org/publications/10003176/fake-account-detection-in-twitter-based-on-minimum-weighted-feature-set) tinha preferência.

### Análise de Sentimentos

A análise de sentimentos foi feita baseando-se na lista de [Benevenuto et al. (2010)](http://www.decom.ufop.br/fabricio/download/ceas10.pdf), onde uma lista de palavras era associada a um sentimento, a lista é a seguinte:

| Categoria                        | Palavras-chave                                  |
| ---------------------------------| ------------------------------------------------|
| Emotion: fear/anxiety            | anxiety/anxious, catastrophic, concern, disaster, emergency, fear, insecure, panic, scared, terror, threat, trouble, warning, worry                 |
| Emotion: shock                   | (taken) aback, floor, god bless, omg, shock, stun, sudden, wtf, wth |
| Response                         | act, asap, escape, evacuate, flee, help, hide, run |
| Need for information and updates | breaking news, call, foul play, incident, phone, report, situation, unconfirmed |
| Assessment: threats              | accident, attack, bomb, bullet, collapse, crash, explode/explosion, fire, gun, hijack, hit, hostage, plan, responsability/responsible, rifle, shot/shoot, struck, suicide, terrorist |
| Assessment: causalities          | blood, body/bodies, corpses, dead, injury/injure, kill, wounded |
| Response and law enforcement     | action, ambulance, command, medic, operation, planes, police/cops/FBI/security, recover, rescue, response, response, safe, safety, save, shut, stay, survive, suspended |
  
Caso a palavra-chave estivesse no tweet, o tweet era "marcado" com aquele sentimento. Um tweet pode expressar múltiplos sentimentos.

### Instalação

> Nota: Os seguintes comandos funcionam em Linux e macOS. Se você está usando Windows, os passos são os mesmos, porém os comandos podem ser diferentes.  

* Baixe e instale Python 2.7+ ou 3.6+ do [site oficial](https://www.python.org/downloads/).  
* Baixe o projeto e extraia seu conteúdo, você deve ver uma pasta chamada `Spammer-Detection-and-Sentiment-Analysis`.  
* Baixe a base de dados [aqui](https://mega.nz/#!s7RjwQoI!wGrWRxv-YTj8hLgIh1LZRl-kHfquIbUtrYi6H1VQB-0) (46GB).
* No Terminal ou Prompt de Comando, acesse a pasta onde você extraiu o projeto.  
* Copie a pasta Spammer-Detection-and-Sentiment-Analysis para sua pasta Home:  
```$ cp -R Spammer-Detection-and-Sentiment-Analysis ~/```  
* Acesse a pasta Home:  
```$ cd ~/```  
* Instale o módulo Virtualenv para isolar o código:  
```$ pip3 install virtualenv```  
* Crie o virtual environment:   
```virtualenv Spammer-Detection-and-Sentiment-Analysis```  
* Entre na pasta do projeto:  
```$ cd Spammer-Detection-and-Sentiment-Analysis```  
* Ative o virtual environment:  
```$ source bin/activate```  
* Instale os módulos do Python:  
```$ pip3 install -r requirements.txt``` 

### Execução das Análises
> Nota: Certifique-se de estar rodando em uma máquina com Memória RAM de no mínimo 16GB (Linux) ou 32GB (macOS).  

Python 2.7+:  
```python bash.py```  

Python 3.5+:  
```python3 bash.py```  

### Documentação

[Clique aqui para ler a documentação](DOCS_PT.md)

### Arquivos

**data/classification**: Classificação dos tweets usados para treinamento dos classificadores, sendo "SPAM" ou "NAO SPAM".   
**data/clean_cl_data**:  1.200 tweets usados para o treinamento dos classificadores, após sua limpeza, ou seja, retirada de caracteres inválidos e problemas de encoding.  
**data/clean_cl_labels**: Classificação dos tweets usados para treinamento dos classificadores, após sua limpeza, ou seja, retirada de caracteres inválidos e problemas de encoding.  
**data/trained_data**: 1.200 tweets usados para o treinamento dos classificadores.  
**remove_training**: Retira do arquivo com os dados os tweets usados no treinamento.    
**quebralinhas**: Quebra o arquivo com os dados em 15 arquivos, com 500 mil tweets cada, este número foi definido baseando-se no tamanho médio que o arquivo de 500 mil tweets tinha. Deveria ser um tamanho adequado para o servidor usado.  
**bash**: Gerencia a fila de arquivos, fazendo com que o processamento de cada arquivo seja feito de forma sequencial.  
**spammer_detection_and_sentiment_analysis**: Faz a análise de cada tweet para detectar spam e fazer análise de sentimentos.   
**paris.txt** (externo): Base de dados com 7.189.160 tweets, relacionados aos ataques terroristas de 13 de novembro de 2015, em Paris, na França. A base de dados completa pode ser obtida [aqui](https://mega.nz/#!s7RjwQoI!wGrWRxv-YTj8hLgIh1LZRl-kHfquIbUtrYi6H1VQB-0) (46GB).   

### Resultado

Da base de 7.189.160 tweets, foram retirados os 1.200 tweets utilizados para treinamento e 1.516 tweets com conteúdo inválido para o algoritmo, restando 7.186.444 tweets.

A classificação dos algorítmos em relação aos tweets serem ou não SPAM indicou alto índice de SPAM, conforme tabela abaixo:

| Método                       | SPAM        | % SPAM   | Não SPAM      | % Não SPAM |
| ---------------------------- | ----------: | :------: | ------------: | :--------: |
| Naïve Bayes Bernoulli        | 1.366.250   | 19%      | 5.820.194     | 81%        |
| Naïve Bayes Multinomial      | 491.120     | 6,8%     | 6.695.324     | 93,2%      |
| Decision Tree                | 1.262.236   | 17,6%    | 5.924.208     | 82,4%      |
| Critérios de El Azab et al.  | 2.043.035   | 28,4%    | 5.143.409     | 71,6%      |
| **Final**                    | **657.476** | **9,1%** | **6.528.968** | **90,9%**  |

A análise de sentimentos identificou todos os sentimentos contidos na tabela de [Benevenuto et al. (2010)](http://www.decom.ufop.br/fabricio/download/ceas10.pdf), o que confirma que sensações ruins relacionadas aos ataques foram geradas nas populações francesa e mundial.

### Problemas e Contribuições

Vários problemas foram identificados após o fim do projeto e podem ser encontrados na seção [issues](https://github.com/amirelemam/Spammer-Detection-and-Sentiment-Analysis/issues). Contribuições são muito bem-vindas.

### Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE)
