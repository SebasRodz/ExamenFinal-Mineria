import pandas as pd
import numpy as np
from hmmlearn import hmm
import streamlit as st

data = pd.read_csv('data_final_covid.csv')

def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
 
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1] @ a[:, j] * b[j, V[t]]
    return alpha

def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
 
    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
 
    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]) @ a[j, :]
 
    return beta

def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        ###estimation step
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            # joint probab of observed data up to time t @ transition prob * emisssion prob as t+1 @
            # joint probab of observed data from time t+1
            denominator = (alpha[t, :].T @ a * b[:, V[t + 1]].T) @ beta[t + 1, :]
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        ### maximization step
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b

def Agregar(A, B, dataset):
    dataset = dataset.append({
        "Status": A,
        "Covid": B
    }, ignore_index=True)
    
    dataset.to_csv('data_final_covid.csv', index=None)
    st.subheader('Datos agregados al final:')
    st.write(dataset.tail(5))
        

def main():
    st.title('Grupo 5 - Míneria de Datos - Examen Final')
    st.text('Se realizara un ejemplo del Modelo Oculto de Markov - BaumWelch')
    
    st.subheader('Dataset a utilizar')
    st.text('El dataset provee de dos columna, donde la primera determina:\n- A: Estado del paciente\n- B: ¿Sigue con Covid?\nMientras que la segunda columna determina:\n- 0: Negación\n- 1: Confirmación\n- 2: No se sabe')
    
    st.write(data.head(30))
    
    st.subheader('Aplicación del Modelo')
    st.text('En esta parte se aplica el modelo hacia los datos definidos.')
    
    V = data['Covid'].values
    
    a = np.ones((2, 2))
    a = a / np.sum(a, axis=1)

    b = np.array(((1, 3, 5), (2, 4, 6)))
    b = b / np.sum(b, axis=1).reshape((-1, 1))

    initial_distribution = np.array((0.5, 0.5))
    n_iter = 100
    a_model, b_model = baum_welch(V.copy(), a.copy(), b.copy(), initial_distribution.copy(), n_iter=n_iter)
    st.text(f'Prediccion para A es \n{a_model} \n \nPrediccion para B es \n{b_model}')
    
    st.subheader('Entrenamiento y Prueba')
    st.text('Para probar el modelo, se comparara con otro llamado hmmlearn.')
    
    model = hmm.MultinomialHMM(n_components=2, n_iter=n_iter, init_params="")
    model.startprob_ = initial_distribution
    model.transmat_ = a
    model.emissionprob_ = b

    model.fit([V])
    #Prediccion
    Z = model.predict([V])
    st.text('Datos de prediccion')
    st.write(Z)

    st.text('Aplicacion de hmmlearn a los datos')
    st.text(f'hmmlearn para A \n{model.transmat_}')
    st.text(f'hmmlearn para B \n{model.emissionprob_}')
    
    st.subheader('Resultados de la implementacion con markov')
    st.text(np.allclose(a_model, model.transmat_, atol=0.1))
    st.text(np.allclose(b_model, model.emissionprob_, atol=0.1))
    
    st.subheader('Comparacion con modelo hmmlearn y markov')
    st.text(f'Diferencia de implementacion de A como ejemplo y hmmlearn \n{a_model - (model.transmat_)}')
    st.text(f'Diferencia de implementacion de B como ejemplo y hmmlearn \n{b_model - (model.emissionprob_)}')
    st.text(f'\nEjemplo de implementacion con A y hmmlear cerrado: {np.allclose(a_model, model.transmat_, atol=0.1)}')
    st.text(f'Ejemplo de implementacion con B y hmmlear cerrado: {np.allclose(b_model, model.emissionprob_, atol=0.1)}')
    
    # Formulario
    st.subheader('Formulario')
    option = ['A', 'B']
    option2 = ['0', '1', '2']
    sentenceAB = st.selectbox('¿Estado del Paciente o Presencia de Covid?', option) 
    sentenceNum = st.selectbox('Cclasificación del atributo anterior:', option2) 
    
    if st.button('Agregar'):
        Agregar(sentenceAB, int(sentenceNum), data)
    
if __name__ == '__main__':
    main()

