# Instalar paquetes (si es necesario)

install.packages("MCMCglmm")

# Cargar paquete

library(MCMCglmm)

# Para este ejemplo usaremos el siguiente dataframe improvisado

datos <- data.frame(
  Estado = factor(c("Sano",
                    "Tos_seca",
                    "Tos_productiva",
                    "Sano",
                    "Tos_seca",
                    "Tos_productiva",
                    "Sano",
                    "Tos_seca",
                    "Tos_productiva",
                    "Sano")),
  Edad = c(35, 62, 48, 27, 55, 71, 41, 38, 66, 50),
  Sexo = factor(c("M", "F", "M", "F", "F", "M", "F", "M", "F", "M")),
  Fumador = factor(c("No", "Si", "Si", "No", "Si", "Si", "No", "No", "Si", "No")),
  Hospital = factor(c("H1", "H1", "H2", "H2", "H3", "H1", "H3", "H2", "H3", "H1"))
)

# Los "prior" son necesarios para cada efecto aleatorio y fijo en la libreria MCMCglmm.
# Será la parte mas dificil de definir, y son cuantificaciones matematicas de lo que 
# nuestra media o varianza de un parametro puede ser.  

prior <- list(
  R = list(V = diag(2), fix = 1), # Con fix, fijamos una categoría aleatoria.
  G = list(
    G1 = list(V = diag(2), nu = 2) # "nu" hace referencia a la distribución de la curva.
  )
)

# Supongamos que:
# Estado = variable categórica con 3 categorías
# Fumador = efecto fijo
# Edad = covariable
# Sexo = Covariable
# Hospital = efecto aleatorio

modelo <- MCMCglmm(
  Estado ~ trait - 1 +
    trait:(Edad + Sexo + Fumador), # Importante especificar "trait" para variables multinomiales.
  random = ~ us(trait):Hospital, 
  rcov = ~ us(trait):units, 
  family = "categorical",
  data = datos,
  prior = prior,
  nitt = 65000,
  burnin = 15000,
  thin = 50
)

# Resumen del modelo

summary(modelo)

# Estadísticas.

autocorr.diag(modelo$Sol)
heidel.diag(modelo$Sol)
