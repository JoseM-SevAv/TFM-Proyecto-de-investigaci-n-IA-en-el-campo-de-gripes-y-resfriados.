install.packages('lme4')
library(lme4)

df <- read.csv("data/datos.csv")

# Creamos un modelo mixto generalizado con las categorias tos seca y tos productiva 
# (tos_seca y size tos_seca respectivamente). Especificamos efectos fijos (period) y 
# aleatorios (1 | herd), para este caso usamos una clasificación binomial

gm1 <- glmer(cbind(tos_seca, size - tos_seca) ~ period + (1 | herd),
             data = df, family = binomial)

summary(gm1)