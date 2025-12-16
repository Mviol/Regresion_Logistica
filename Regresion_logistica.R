# paquetes usados

install.packages(c("titanic", "dlookr", "questionr", "sjPlot", "car", "effects", "caret", "pROC"))

# Librerías cargadas 

library(titanic)        # Para obtener el dataset del Titanic
library(dlookr)         # Para análisis de correlación y visualización
library(questionr)      # Para calcular odds ratios
library(sjPlot)         # Para visualización de modelos y tablas
library(car)            # Para diagnóstico de modelos (VIF, outliers)
library(effects)        # Para gráficos de efectos
library(caret)          # Para matriz de confusión y métricas
library(pROC)           # Para curvas ROC y cálculo de AUC


# Carga la biblioteca que contiene los datos del Titanic
library(titanic)

# Asigna nombres a los subconjuntos de datos
# train: conjunto de entrenamiento
# test: conjunto de prueba
train <- titanic_train
test <- titanic_test

# Combina el conjunto de prueba con las predicciones del modelo de género-clase
# Esto añade la variable 'Survived' al conjunto de prueba
test <- merge(test, titanic_gender_class_model, by="PassengerId")

# Verifica la estructura de las variables en el conjunto de entrenamiento
str(train)

# Variables del dataset:
# Survived: 0 = No sobrevivió, 1 = Sobrevivió
# SibSp: Número de hermanos/cónyuges a bordo
# Parch: Número de padres/hijos a bordo
# Fare: Tarifa pagada
# Embarked: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)
# Pclass: Clase del boleto (1ra, 2da, 3ra clase)

# Verifica valores faltantes (NA) en ambos conjuntos
colSums(is.na(train))
colSums(is.na(test))

# Verifica valores vacíos (cadenas vacías)
colSums(train == '')
colSums(test == '')

# Elimina filas con valores faltantes o vacíos
test <- test[-which(is.na(test$Fare)),]      # Elimina filas con Fare = NA
train <- train[-which(train$Embarked == ""),] # Elimina filas con Embarked vacío

# Imputación de valores faltantes en la variable Age
# Usa la mediana como estrategia básica de imputación
train$Age[is.na(train$Age)] <- median(train$Age, na.rm=T)
test$Age[is.na(test$Age)] <- median(test$Age, na.rm=T)

# Elimina variables consideradas irrelevantes para el modelo
train <- subset(train, select = -c(Cabin, PassengerId, Ticket, Name))
test <- subset(test, select = -c(Cabin, PassengerId, Ticket, Name))

# Convierte variables categóricas a tipo factor
for (i in c("Survived","Pclass","Sex","Embarked")){
  train[,i] <- as.factor(train[,i])
}
for (j in c("Survived","Pclass","Sex","Embarked")){
  test[,j] <- as.factor(test[,j])
}

# Variable respuesta: Survived
# Variables explicativas: todas las demás

# Analiza correlaciones entre variables numéricas
library(dlookr)
correlate(train)        # Calcula matriz de correlaciones
plot_correlate(train)   # Visualiza matriz de correlaciones

# Elimina filas con datos incompletos (por si acaso)
train <- train[complete.cases(train),]

# Verifica balance de clases en la variable respuesta
table(train$Survived)               # Conteo absoluto
prop.table(table(train$Survived))   # Proporciones
# Resultado: dataset levemente desbalanceado

# Modelo 1: Incluye todas las variables
mod1 <- glm(Survived ~ ., data = train, family = binomial(link = "logit"))

# Interpretación del summary:
# - Null deviance: mide cómo la variable respuesta es predicha solo por el intercepto
# - Residual deviance: mide cómo la variable respuesta es predicha por el modelo completo
# - AIC: medida de ajuste que penaliza por número de variables (menor es mejor)

# Test de Razón de Verosimilitud
anova(mod1, test="Chisq")  # Evalúa mejora al añadir variables secuencialmente
drop1(mod1, test="Chisq")  # Evalúa impacto de remover variables secuencialmente

# Modelo 2: Modelo reducido basado en análisis preliminar
mod2 <- glm(Survived ~ Pclass + Sex + Age + SibSp,
            data = train, family = binomial(link = "logit"))

# Comparación entre modelos usando Test de Razón de Verosimilitud
# Si p > nivel de significancia, las variables omitidas no son significativas
anova(mod2, mod1, test="LRT")

# Intervalos de confianza para los coeficientes
confint(mod2)

# Selección automática de variables usando método backward (basado en AIC)
mod3 <- step(mod1, direction = "backward")

# Comparación entre el modelo automático y el modelo reducido manual
anova(mod3, mod2, test="LRT")  # Si p > 0.05, Embarked puede excluirse

# Cálculo de Odds Ratios (Razón de Momios)
library(questionr)
odds.ratio(mod2)  # Muestra el efecto de cada variable en términos de odds ratio

# Visualización de coeficientes del modelo
library(sjPlot)
plot_model(mod2, vline.color = "red", sort.est = TRUE, 
           show.values = TRUE, value.offset = .3)

# Evaluación basada en Deviance:
# Null deviance = 2(LL(modelo saturado) - LL(modelo nulo))
# Residual deviance = 2(LL(modelo saturado) - LL(modelo propuesto))
# Criterio: Residual deviance/(n-k) < 1 indica modelo adecuado
# En este caso: 790.68/(889-7) < 1 → Modelo ADECUADO

# Comparación de AIC entre modelos (menor es mejor)
AIC(mod1)
AIC(mod2)
AIC(mod3)

# Gráficos marginales: comparan valores observados vs ajustados
marginalModelPlots(mod2)

# Detección de outliers
car::outlierTest(mod2)  # Test de Bonferroni para outliers

# Identificación de puntos influentes
influenceIndexPlot(mod2)  # Múltiples gráficos de influencia
influencePlot(mod2, col = "red", id = list(method = "noteworthy", 
                                           n = 4, cex = 1, col = carPalette()[1], 
                                           location = "lr"))
# Valores que exceden ±2 en distancia de Cook: 262, 631, 298, etc.

# Verificación del impacto de puntos potencialmente influentes
mod2_298 <- update(mod2, subset = c(-298))  # Remueve observación 298
car::compareCoefs(mod2, mod2_298)  # Compara coeficientes

# Diagnóstico de multicolinealidad usando VIF (Variance Inflation Factor)
library(car)
vif(mod2)  # Valores < 5 indican ausencia de multicolinealidad severa

# Gráficos de efectos para visualizar relaciones
library(effects)
plot(allEffects(mod2))

# Predicciones en el conjunto de prueba
pred <- predict(mod2, test, type = "response")  # Probabilidades predichas
result <- as.factor(ifelse(pred > 0.5,1,0))      # Clasificación con umbral 0.5

# Evaluación usando matriz de confusión
library(caret)
confusionMatrix(result, test$Survived, positive = "1")

# Análisis ROC y AUC
library(pROC)
auc <- roc(test$Survived, pred)  # Calcula curva ROC
plot.roc(auc, print.thres = T)   # Muestra curva y umbral óptimo
# El umbral óptimo es 0.551 (maximiza sensibilidad + especificidad)

# Clasificación con nuevo umbral óptimo
result2 <- as.factor(ifelse(pred > 0.551,1,0))
confusionMatrix(result2, test$Survived, positive = "1")

# Genera tabla formateada del modelo para reportes
library(sjPlot)
tab_model(mod2)

