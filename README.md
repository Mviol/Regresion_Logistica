ESPAÑOL
📋 DESCRIPCIÓN GENERAL
Este script de R realiza un análisis completo de regresión logística utilizando el famoso dataset del Titanic. El objetivo es predecir si un pasajero sobrevivió (1) o no (0) basándose en características como clase, sexo, edad y familiares a bordo.

🔧 FLUJO DE TRABAJO
1. PREPARACIÓN DE DATOS
Carga y combina los datasets de entrenamiento y prueba del Titanic

Limpia datos faltantes (NA) y valores vacíos

Imputa la edad usando la mediana cuando falta

Elimina variables irrelevantes (Cabin, PassengerId, Ticket, Name)

Convierte variables categóricas a factores

2. ANÁLISIS EXPLORATORIO
Examina correlaciones entre variables numéricas

Verifica el balance de clases (sobrevivientes vs no sobrevivientes)

Genera visualizaciones de correlación

3. MODELADO ESTADÍSTICO
Modelo 1: Incluye todas las variables disponibles

Modelo 2: Modelo simplificado con variables más significativas (Pclass, Sex, Age, SibSp)

Modelo 3: Selección automática de variables usando método backward

4. EVALUACIÓN DEL MODELO
Compara modelos usando tests de razón de verosimilitud

Calcula intervalos de confianza para coeficientes

Evalúa calidad del ajuste con deviance y AIC

Calcula odds ratios para interpretación de efectos

5. DIAGNÓSTICOS
Detecta outliers y puntos influyentes

Verifica supuesto de multicolinealidad (VIF)

Genera gráficos de efectos marginales

Analiza residuos del modelo

6. PREDICCIÓN Y VALIDACIÓN
Predice probabilidades en datos de prueba

Aplica umbral de clasificación (0.5 inicial, 0.551 óptimo)

Evalúa desempeño con matriz de confusión

Calcula curva ROC y área bajo la curva (AUC)

7. REPORTE
Genera tablas formateadas para publicación

Visualiza coeficientes del modelo

Exporta resultados interpretables

🎯 RESULTADOS CLAVE
Variables significativas: Clase, sexo, edad y número de hermanos/cónyuges

Mejor modelo: Modelo 2 (balance entre simplicidad y poder predictivo)

Umbral óptimo: 0.551 para clasificación

Supuestos cumplidos: Sin multicolinealidad severa, pocos outliers influyentes

📊 APLICACIONES PRÁCTICAS
Ejemplo educativo de regresión logística completa

Plantilla reutilizable para análisis similares

Demostración de buenas prácticas en modelado predictivo

Base para proyectos de ciencia de datos con R

ENGLISH
📋 OVERVIEW
This R script performs a comprehensive logistic regression analysis using the famous Titanic dataset. The goal is to predict whether a passenger survived (1) or not (0) based on characteristics such as class, gender, age, and family members aboard.

🔧 WORKFLOW
1. DATA PREPARATION
Loads and combines Titanic training and test datasets

Cleans missing values (NA) and empty strings

Imputes age using median when missing

Removes irrelevant variables (Cabin, PassengerId, Ticket, Name)

Converts categorical variables to factors

2. EXPLORATORY ANALYSIS
Examines correlations between numerical variables

Checks class balance (survivors vs non-survivors)

Generates correlation visualizations

3. STATISTICAL MODELING
Model 1: Includes all available variables

Model 2: Simplified model with most significant variables (Pclass, Sex, Age, SibSp)

Model 3: Automatic variable selection using backward method

4. MODEL EVALUATION
Compares models using likelihood ratio tests

Calculates confidence intervals for coefficients

Evaluates model fit with deviance and AIC

Computes odds ratios for effect interpretation

5. DIAGNOSTICS
Detects outliers and influential points

Checks multicollinearity assumption (VIF)

Generates marginal effects plots

Analyzes model residuals

6. PREDICTION AND VALIDATION
Predicts probabilities on test data

Applies classification threshold (0.5 initial, 0.551 optimal)

Evaluates performance with confusion matrix

Calculates ROC curve and area under curve (AUC)

7. REPORTING
Generates formatted tables for publication

Visualizes model coefficients

Exports interpretable results

🎯 KEY RESULTS
Significant variables: Class, gender, age, and number of siblings/spouses

Best model: Model 2 (balance between simplicity and predictive power)

Optimal threshold: 0.551 for classification

Assumptions met: No severe multicollinearity, few influential outliers

📊 PRACTICAL APPLICATIONS
Educational example of complete logistic regression

Reusable template for similar analyses

Demonstration of good practices in predictive modeling

Foundation for data science projects with R

ITALIANO
📋 DESCRIZIONE GENERALE
Questo script R esegue un'analisi completa di regressione logistica utilizzando il famoso dataset del Titanic. L'obiettivo è predire se un passeggero è sopravvissuto (1) o no (0) basandosi su caratteristiche come classe, sesso, età e familiari a bordo.

🔧 FLUSSO DI LAVORO
1. PREPARAZIONE DATI
Carica e combina dataset di training e test del Titanic

Pulisce valori mancanti (NA) e stringhe vuote

Imputa l'età usando la mediana quando mancante

Rimuove variabili irrilevanti (Cabin, PassengerId, Ticket, Name)

Converte variabili categoriali in fattori

2. ANALISI ESPLORATIVA
Esamina correlazioni tra variabili numeriche

Verifica bilanciamento classi (sopravvissuti vs non sopravvissuti)

Genera visualizzazioni di correlazione

3. MODELLAZIONE STATISTICA
Modello 1: Include tutte le variabili disponibili

Modello 2: Modello semplificato con variabili più significative (Pclass, Sex, Age, SibSp)

Modello 3: Selezione automatica variabili usando metodo backward

4. VALUTAZIONE MODELLO
Confronta modelli usando test rapporto di verosimiglianza

Calcola intervalli di confidenza per coefficienti

Valuta adattamento modello con devianza e AIC

Calcola odds ratio per interpretazione effetti

5. DIAGNOSTICHE
Rileva outliers e punti influenti

Verifica assunzione multicollinearità (VIF)

Genera grafici effetti marginali

Analizza residui del modello

6. PREDIZIONE E VALIDAZIONE
Predice probabilità su dati di test

Applica soglia classificazione (0.5 iniziale, 0.551 ottimale)

Valuta performance con matrice di confusione

Calcola curva ROC e area sotto curva (AUC)

7. REPORTING
Genera tabelle formattate per pubblicazione

Visualizza coefficienti del modello

Esporta risultati interpretabili

🎯 RISULTATI CHIAVE
Variabili significative: Classe, sesso, età e numero fratelli/coniugi

Miglior modello: Modello 2 (bilancio tra semplicità e potere predittivo)

Soglia ottimale: 0.551 per classificazione

Assunzioni rispettate: Nessuna multicollinearità severa, pochi outliers influenti

📊 APPLICAZIONI PRATICHE
Esempio educativo di regressione logistica completa

Template riutilizzabile per analisi simili

Dimostrazione buone pratiche modellazione predittiva

Base per progetti data science con R

PORTUGUÊS
📋 DESCRIÇÃO GERAL
Este script R realiza uma análise completa de regressão logística utilizando o famoso conjunto de dados do Titanic. O objetivo é prever se um passageiro sobreviveu (1) ou não (0) com base em características como classe, sexo, idade e familiares a bordo.

FLUXO DE TRABALHO
 PREPARAÇÃO DE DADOS
Carrega e combina conjuntos de treinamento e teste do Titanic

Limpa valores faltantes (NA) e strings vazias

Imputa idade usando mediana quando faltante

Remove variáveis irrelevantes (Cabin, PassengerId, Ticket, Name)

Converte variáveis categóricas em fatores

ANÁLISE EXPLORATÓRIA
Examina correlações entre variáveis numéricas

Verifica balanceamento de classes (sobreviventes vs não sobreviventes)

Gera visualizações de correlação

MODELAGEM ESTATÍSTICA
Modelo 1: Inclui todas as variáveis disponíveis

Modelo 2: Modelo simplificado com variáveis mais significativas (Pclass, Sex, Age, SibSp)

Modelo 3: Seleção automática de variáveis usando método backward

AVALIAÇÃO DO MODELO
Compara modelos usando testes de razão de verossimilhança

Calcula intervalos de confiança para coeficientes

Avalia ajuste do modelo com deviance e AIC

Calcula odds ratios para interpretação de efeitos

DIAGNÓSTICOS
Detecta outliers e pontos influentes

Verifica suposição de multicolinearidade (VIF)

Gera gráficos de efeitos marginais

Analisa resíduos do modelo

 PREDIÇÃO E VALIDAÇÃO
Prevê probabilidades em dados de teste

Aplica limiar de classificação (0.5 inicial, 0.551 ótimo)

Avalia desempenho com matriz de confusão

Calcula curva ROC e área sob curva (AUC)

RELATÓRIO
Gera tabelas formatadas para publicação

Visualiza coeficientes do modelo

Exporta resultados interpretáveis

RESULTADOS CHAVE
Variáveis significativas: Classe, sexo, idade e número de irmãos/cônjuges

Melhor modelo: Modelo 2 (equilíbrio entre simplicidade e poder preditivo)

Limiar ótimo: 0.551 para classificação

Suposições atendidas: Sem multicolinearidade severa, poucos outliers influentes

 APLICAÇÕES PRÁTICAS
Exemplo educacional de regressão logística completa

Modelo reutilizável para análises similares

Demonstração de boas práticas em modelagem preditiva

Base para projetos de ciência de dados com R

CONCEITOS-CHAVE EM TODOS OS IDIOMAS
VARIÁVEIS DO MODELO FINAL
Pclass: Classe socioeconômica (1ª, 2ª, 3ª classe)

Sex: Gênero do passageiro

Age: Idade do passageiro

SibSp: Número de irmãos/cônjuges a bordo

MÉTRICAS DE DESEMPENHO
AUC: Área sob curva ROC (poder discriminatório)

Accuracy: Precisão global das predições

Sensitivity/Recall: Capacidade de detectar sobreviventes

Specificity: Capacidade de detectar não-sobreviventes

