import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from pycaret.classification import load_model, predict_model


modelo1 = load_model('Melhor Modelo para Custos')
modelo2 = load_model('meu-melhor-modelo-para-smoker')

def classificador(modelo, dados):
	pred = predict_model(estimator = modelo, data = dados)
	return pred

def smap(x):
	y = 'masculine' if x == 'Masculino' else 'feminine'
	return y

def rmap(x):
	if x == 'Sudeste':
		return 'southeast'
	elif x == 'Noroeste':
		return 'northwest'
	elif x == 'Sudoeste':
		return 'southwest'
	else:
		return 'northeast'


def fmap(x):
	y = 'yes' if x == 'Sim' else 'no'
	return y


st.sidebar.header('**Medical Cost Deploy Center**')

opcoes = ['Página Inicial',
		  'Cotação do seguro',
		  'Probabilidade de fraude',
		  'Observações']

pagina = st.sidebar.selectbox('Navegue pelo menu abaixo:', opcoes)

st.sidebar.markdown('---')

##### Página Inicial #####

if pagina == 'Página Inicial':
	st.write("""
	# Bem vindo ao Medical Cost Deploy Center
    Nese Web-App podemos aplicar dois modelos desenvolvidos: 
    i) Precificação de novos seguros;
    ii) Busca de possíveis fraudadores do seguro.
    A lista abaixo mostra o que está implentado até o presente momento.
    
    Funcionalidades no momento:
    -[x] Página Inicial
    -[x] Modelo de precificação de planos de saúde para novos clientes
    -[x] Modelo de detecção de prossíveis fraudadores
    -[ ] Deploy em lote (várias pessoas simultaneamente)
    -[x] Página de créditos

    Os modelos desse web-app foram desenvolvidos através do conjunto de dados [link do kaggle](https://www.kaggle.com/mirichoi0218/insurance).
    O referencial sobre os modelos utilizados você pode encontrar em: [link](https://github.com/rodrigoviana-ds/Projetos/blob/main/ModeloFinal.ipynb).
    Caso encontre algum erro ou queira uma explicação, entre em contato: rodrigo_viana@id.uff.br
    Para mais informações sobre o streamlit, consulte [site oficial](https://streamlit.io)
    """)
	st.markdown('---')

##### Página: Cotação do Seguro #####

elif pagina == 'Cotação do seguro':
	st.markdown('![alt text](https://https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide3.jpg?raw=true)')
	
	st.markdown('# Precificação do Valor do Seguro')
	
	st.markdown('Nessa seção é feita a implementação do modelo para cotar o valor do seguro para um indivíduo. Entre com as informações e clique em APLICAR O MODELO para obter a precificação.')

	st.markdown('---')

	idade = st.number_input('Idade', 18, 65, 30)
	imc = st.number_input('Índice de Massa Corporal:', 15, 54, 24)
	sexo = st.selectbox("Sexo:", ['Masculino', 'Feminino'])
	criancas = st.selectbox("Quantidade de filhos:",[ 0, 1, 2, 3, 4, 5])
	fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = st.selectbox("Região em que mora:", ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

	#custos = st.number_input('Custos da Pessoa', 1000, 64000, 10000)

	dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 'children': [criancas], 'region': [rmap(regiao)], 'smoker': [fmap(fumante)]}
	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo1, dados)
		pred = float(saida['Label'].round(2))
		valor = round(1.80*pred, 2)

		s1 = 'Custo Anual Estimado do Seguro: $ {:.2f}'.format(pred)
		s2 =  'Valor de Venda do Seguro: ${:.2f}'.format(valor)

		st.markdown('## Resultados do Modelo para as entradas:')
		st.write(dados)
		st.markdown('## **' + s1 + '**')
		st.markdown('## **' + s2 + '**')


##### Página: Probabilidade de Fraude #####

elif pagina == 'Probabilidade de fraude':
	st.markdown('![alt text](https://https://github.com/ricardorocha86/WebApp-MedicalCost/blob/main/imagens/Slide3.jpg?raw=true)')
	
	st.markdown('# Detectar probabilidade de fraude')
	
	st.markdown('O deploy é feito na variável **fumante**. Insira as informações e clique em APLICAR O MODELO para obter as predições.')
	
	st.markdown('---')

	idade = st.number_input('Idade', 18, 65, 30)
	sexo = st.selectbox("Sexo", ['Masculino', 'Feminino'])
	imc = st.number_input('Índice de Massa Corporal', 15, 54, 24)
	criancas = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5])
	#fumante = st.selectbox("É fumante?", ['Sim', 'Não'])
	regiao = st.selectbox("Região em que mora", ['Sudeste', 'Noroeste', 'Sudoeste', 'Nordeste'])

	custos = st.number_input('Custos da pessoa', 1000, 64000, 10000)

	dados_dicio = {'age': [idade], 'sex': [smap(sexo)], 'bmi': [imc], 'children': [criancas], 'region': [rmap(regiao)], 'charges': [custos]}

	dados = pd.DataFrame(dados_dicio)

	st.markdown('---')

	if st.button('APLICAR O MODELO'):
		saida = classificador(modelo2, dados)
		resp = 'NÃO' if saida['Label'][0] == 'no' else 'SIM'
		prob = saida['Score'][0]
		st.markdown('## **O indivíduo em análise é fumante?**')
		s = 'Resposta do modelo: {}, com probabilidade {:.2f}%.'.format(resp, 100*prob)
		st.markdown('## **' + s + '**')

		if resp == 'NÃO':
			st.success('Baixa probabilidade de fraude!')
		elif prob < 0.7:
			st.warning('Probabilidade moderada de fraude!')
		else:
			st.error('Probabilidade alta de fraude!!!')


##### Página: Observações #####

else:
	st.write(""" # Observações """)

	st.markdown('Nesse Web-App mostro uma aplicação e solução rápida em Data Science. O próximo passo para tornar o modelo ainda melhor, é implementar uma matriz de custo para o caso de um cliente ser classificado como fumante ou não de maneira errada.')

	st.markdown('Tenho muito a me desenvolver e quero ter essa oportunidade junto ao time MAG Seguros.')

	st.markdown('---')

	if st.button('Obrigado'):
		st.balloons()



