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

| Atributos                                        | Peso   |
| :----------------------------------------------: |:------:|
| The account has at least 30 followers            | 0.53   |
| The account has been geo-localized               | 0.85   |
| It has been included in another user's favorites | 0.85   |
| It has used a hashtag in at least one tweet      | 0.85   |
| It has logged into Twitter using an iPhone       | 0.917  |
| A mention by Twitter user                        | 1      |
| It has written at least 50 tweets                | 0.01   |
| It has been included in another user's list      | 0.45   |
| (2*number of followers) _ (number of friends)    | 0.5    |
| User has at least one Favorite list              | 0.17   |

Com os quatro classificadores treinados, o tweet era indicado como SPAM caso a maioria dos classificadores apontasse como tal. Caso houvesse empate, a classificação feita pelos critérios de [El Azab et al. (2016)](http://waset.org/publications/10003176/fake-account-detection-in-twitter-based-on-minimum-weighted-feature-set) tinha preferência.

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
  
Caso a palavra se encontrasse no tweet, o tweet era "marcado" com aquele sentimento. Um tweet podel expressar múltiplos sentimentos.

### Instalação


### Execução das Análises


### Métodos

Segue abaixo a lista com o nome dos métodos e seu papel no código:

```

```
### Arquivos

**remove_training**: Retira do arquivo com os dados os tweets usados no treinamento.   
**quebralinhas**: Quebra o arquivo com os dados em 15 arquivos, com 500 mil tweets cada, este número foi definido baseando-se no tamanho médio que o arquivo de 500 mil tweets tinha. Deveria ser um tamanho adequado para o servidor.  
**bash**: Gerencia a fila de arquivos, fazendo com que o processamento de cada arquivo seja feito de forma sequencial.  
**spammer_detection_and_sentiment_analysis**: Faz a análise de cada tweet para detectar spam e fazer análise de sentimentos.   
**paris.txt** (externo): Base de dados com 7.189.160 tweets, relacionados aos ataques terroristas de 13 de novembro de 2015, em Paris, na França. A base de dados completa (46GB) pode ser obtida [aqui](), no [Mirror 1]() e no [Mirror 2]().   

### Resultado

Da base de 7.189.160 tweets, foram retirados os 1.200 tweets utilizados para treinamento e 1.516 tweets com conteúdo inválido para o algoritmo, restando 7.186.444 tweets.

A classificação dos algorítmos em relação aos tweets serem ou não SPAM indicou alto índice de SPAM, conforme tabela abaixo:

| Método                       | SPAM        | % SPAM   | Não SPAM      | % Não SPAM |
| ---------------------------- | ----------: | -------: | ------------: | ---------: |
| Naïve Bayes Bernoulli        | 1.366.250   | 19%      | 5.820.194     | 81%        |
| Naïve Bayes Multinomial      | 491.120     | 6,8%     | 6.695.324     | 93,2%      |
| Decision Tree                | 1.262.236   | 17,6%    | 5.924.208     | 82,4%      |
| Critérios de El Azab et al.  | 2.043.035   | 28,4%    | 5.143.409     | 71,6%      |
| **Overall**                  | **657.476** | **9,1%** | **6.528.968** | **90,9%**  |

A análise de sentimentos identificou todos os sentimentos contidos na tabela de [Benevenuto et al. (2010)](http://www.decom.ufop.br/fabricio/download/ceas10.pdf), o que confirma que sensações ruins relacionadas aos ataques foram geradas nas populações francesa e mundial.

### Problemas

Vários problemas foram identificados após o fim do projeto e podem ser encontrados na seção `issues`. Contribuições são muito bem-vindas.

### Licença

Estes arquivos estão licenciados sob a [Licença MIT](LICENSE)
