################################################################################################
#
# MODELAGEM PREDITIVA - MBA Business Analytics e Big Data
# Por: RICARDO REIS
#
# CASE - MUSHROOM CLASSIFICATION
#
################################################################################################

# LENDO OS DADOS
path <- "C:/Users/Ricardo/Documents/R-Projetos/MushroomClassification/"
base <- read.csv(paste0(path, "dataset-mushrooms.csv"), sep = ",")

summary(base)

library(VIM)
matrixplot(base)
aggr(base)

################################################################################################
# AMOSTRAGEM DO DADOS
library(caret)
set.seed(12345)
index <- createDataPartition(base$class, p= 0.7,list = F)
data.train <- base[index, ] # base de desenvolvimento: 70%
data.test  <- base[-index,] # base de teste: 30%

# Checando se as proporções das amostras são próximas à base original
prop.table(table(base$class))
prop.table(table(data.train$class))
prop.table(table(data.test$class))

# Algoritmos de árvore necessitam que a variável resposta num problema de classificação seja 
# um factor; convertendo aqui nas amostras de desenvolvimento e teste
data.train$class <- as.factor(data.train$class)
data.test$class  <- as.factor(data.test$class)

################################################################################################
# MODELAGEM DOS DADOS - RANDOM FOREST

names  <- names(data.train) # salva o nome de todas as variáveis e escreve a fórmula
f_full <- as.formula(paste("class ~",
                           paste(names[!names %in% "class"], collapse = " + ")))
library(randomForest)
rndfor <- randomForest(f_full,data= data.train,importance = T, nodesize =200, ntree = 500)
rndfor

# Avaliando a evolução do erro com o aumento do número de árvores no ensemble
plot(rndfor, main= "Mensuração do erro")
legend("topright", c('Out-of-bag',"1","0"), lty=1, col=c("black","green","red"))

# Uma avaliação objetiva indica que a partir de ~100 árvores não há mais ganhos expressivos
rndfor2 <- randomForest(f_full,data= data.train,importance = T, nodesize =2500, mtry =2, ntree = 100)
rndfor2

plot(rndfor2, main= "Mensuração do erro")
legend("topright", c('Out-of-bag',"1","0"), lty=1, col=c("black","green","red"))

# Importância das variáveis
varImpPlot(rndfor2, sort= T, main = "Importância das Variáveis")

# Aplicando o modelo nas amostras  e determinando as probabilidades
rndfor2.prob.train <- predict(rndfor2, type = "prob")[,2]
rndfor2.prob.test  <- predict(rndfor2,newdata = data.test, type = "prob")[,2]

# Comportamento da saida do modelo
hist(rndfor2.prob.test, breaks = 25, col = "lightblue",xlab= "Probabilidades",
     ylab= "Frequência",main= "Random Forest")
boxplot(rndfor2.prob.test ~ data.test$class,col= c("green", "red"), horizontal= T)

################################################################################################
# AVALIANDO A PERFORMANCE

# Métricas de discriminação para ambos modelos
library(hmeasure) 

rndfor.train  <- HMeasure(data.train$class,rndfor2.prob.train)
rndfor.test  <- HMeasure(data.test$class,rndfor2.prob.test)
rndfor.train$metrics
rndfor.test$metrics

library(pROC)

roc <- roc(data.test$class,rndfor2.prob.test)
y <- roc$sensitivities
x <- 1-roc$specificities

plot(x,y, type="n",
     xlab = "1 - Especificidade", 
     ylab= "Sensitividade")
lines(x, y,lwd=3,lty=1, col="purple") 
legend("bottomright", c('Random Forest'), lty=1, col=c("purple"))

################################################################################################
################################################################################################
