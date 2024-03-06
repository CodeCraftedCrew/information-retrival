# Sistema de Recuperación de Información

Nahomi Bouza Rodriguez C412
Yisell Martinez Noa C412

## Descripción del modelo

### Booleano

$D$ Conjunto de términos indexados.   
$Q$ Expresion booleana sobre los términos indexados utilizando las operaciones: not, and, or.  
$F$ Álgebra booleana sobre conjunto de términos y conjuntos de documentos.  
$R$ Función booleana que indica si el documento $d_j$ es relevante a la consulta $q$.  

En este modelo los pesos de términos indexados son binarios $w_{ij} \in {0,1}$. Una consulta $q$ es una expresión booleana convencional. Sea $\vec{q_{fnd}}$ la Forma Normal Disyuntiva de la consulta $q$ y $\vec{q_{cc}}$ una de las componentes conjuntivas de $\vec{q_{fnd}}$. La similutud entre un documento $\vec{d_j}$ y la consulta q se define como:

$sim(d_j, q) = \{ 1 \; si \;  \exists \vec{q_{cc}}: (\vec{q_{cc}} \in \vec{q_{fnd}}) \land(\forall k_i, g_i (\vec{d_j}) = g_i(\vec{g_{cc}})), 0 \; en \; otro \; caso \;\}$

#### Especificaciones

Usamos las herramientas vistas en clase práctica para el pre-procesamiento de los documentos y queries.

El texto se somete a una tokenización utilizando la biblioteca Spacy, con el objetivo de dividir el texto en unidades significativas. Posteriormente, se lleva a cabo una fase de eliminación de ruido para eliminar caracteres irrelevantes o no deseados. A continuación, se procede a la eliminación de las palabras vacías (stopwords) para reducir la dimensionalidad del texto y concentrarse en las palabras clave. Además, se realiza una normalización de las letras, convirtiendo todas las letras a minúsculas para garantizar la coherencia en el análisis. Finalmente, se lleva a cabo la separación en formas normales disyuntivas para homogeneizar la estructura del texto y facilitar su procesamiento posterior.

Una vez terminado este pre-procesamiento se determina la relevancia de un documento en función de la cantidad de términos coincidentes entre el documento y la consulta realizada.

Primero, se calcula el número mínimo de coincidencias requeridas para que un documento sea considerado relevante, utilizando un umbral de relajación. Este umbral se basa en el porcentaje del tamaño de la consulta, donde un valor de 0 indica una coincidencia exacta con la consulta, y valores mayores permiten una flexibilidad en la coincidencia.

Luego, se verifica cuántos términos de la consulta están presentes en el documento. Si esta cantidad supera o iguala el número mínimo de coincidencias requeridas, el documento se considera relevante y se incluye en la lista de documentos relevantes.

#### Consideraciones

Se puede realizar la consulta sobre cualquier corpus disponible para carga con ir_datasets, pues se tiene como parámetro del modelo el nombre del dataset.

Nuestro sistema permite que las consultas sean introducidas en lenguaje natural. Los usuarios pueden solicitudes en forma de oraciones, y el programa las procesará para recuperar los documentos que cumplan con los criterios de la consulta.

### Booleano extendido

$D$ Conjunto de términos indexados
$Q$ Expresion booleana sobre los términos indexados utilizando las operaciones: not, and, or
$F$ Álgebra booleana sobre conjunto de términos y conjuntos de documentos.
$R$ Función booleana que indica si el documento $d_j$ es relevante a la consulta $q$.

En este modelo los pesos de términos indexados $w_{ij} \in [0,1]$. Una consulta $q$ es una expresión booleana convencional. Sea $\vec{q_{fnd}}$ la Forma Normal Disyuntiva de la consulta q y $\vec{q_{cci}}$ una de las componentes conjuntivas de $\vec{q_{fnd}}$. La similutud entre un documento $\vec{d_j}$ y la consulta q se define como:

$sim(q_{andi}, dj) = 1 - \sqrt[p]{\frac{\sum_{i=1}^{t}(1-w_i)^p}{t}}$

$sim(q_{or}, dj) = \sqrt[p]{\frac{\sum_{i=1}^{t}w_i^p}{t}}$

siendo $w_i= f_{x,j}*\frac{Idf_x}{max_iIdf_x}$

Así, la similitud del documento con la consulta se calcula como la similitud $or$ de los resultados de las similitudes $and$ para cada una de las formas disyuntivas.

#### Especificaciones

Se usan las mismas herramientas de pre-procesamiento descritas anteriormente para el modelo booleano.

Este código implementa un modelo de recuperación de información extendido basado en consultas booleanas. A diferencia de los modelos booleanos tradicionales, que evalúan consultas utilizando operadores lógicos simples como AND, OR y NOT, este modelo extendido utiliza una función de similitud para calcular la relevancia de los documentos en función de los términos de la consulta y sus pesos asociados.

Se utiliza una función de similitud para comparar los pesos de los términos de la consulta con los pesos del documento y determinar si el documento es relevante para la consulta dada. Este enfoque permite una mayor flexibilidad en la evaluación de la relevancia de los documentos, ya que los pesos pueden reflejar la importancia relativa de los términos en la consulta y en el documento.

### Métricas

Apoyados en lo estudiado en clase práctica desarollamos la clase Metrics, que se utiliza para calcular diferentes métricas de evaluación de un sistema de recuperación de información.

**Cálculo de Documentos Relevantes:** La clase puede identificar los documentos relevantes para una consulta específica. Esto se logra comparando la consulta con los documentos anotados en un conjunto de datos de relevancia (qrels). Los documentos marcados como relevantes se seleccionan según un nivel de relevancia específico (3 o 4).

**Cálculo de Métricas de Evaluación:** La clase proporciona métodos para calcular varias métricas de evaluación de recuperación de información, incluyendo precisión, recall, puntaje F, R-precisión y fallout. Estas métricas se utilizan para evaluar el rendimiento de un modelo de recuperación de información en función de los documentos recuperados y los documentos relevantes para una consulta dada.

1. Precisión (Precision):

    La precisión se refiere a la proporción de documentos recuperados que son relevantes para la consulta.
    **Interpretación:** Un valor alto de precisión indica que la mayoría de los documentos recuperados son relevantes para la consulta. Es útil cuando se desea minimizar el número de documentos irrelevantes en los resultados.

2. Recall (Recall):

    El recall (también conocido como sensibilidad) se refiere a la proporción de documentos relevantes que fueron recuperados correctamente.
    **Interpretación:** Un valor alto de recall indica que la mayoría de los documentos relevantes fueron recuperados. Es útil cuando se desea maximizar la recuperación de documentos relevantes, incluso si eso significa recuperar algunos documentos irrelevantes adicionales.

3. Puntaje F (F1 Score):

    El puntaje F es una medida que combina precisión y recall en un solo valor. Se calcula como la media armónica de precisión y recall.
    **Interpretación:** Un valor alto de puntaje F indica un buen equilibrio entre precisión y recall. Es útil cuando se desea una métrica que tenga en cuenta tanto la precisión como la exhaustividad del sistema de recuperación de información.

4. R-Precisión (R-Precision):

    La R-precisión se refiere a la precisión calculada considerando los primeros R documentos recuperados, donde R es el número total de documentos relevantes para la consulta.
    **Interpretación:** La R-precisión proporciona una medida de la precisión del sistema al considerar solo los primeros documentos recuperados. Es útil cuando se desea evaluar el rendimiento del sistema en los primeros resultados, que son los más relevantes para el usuario.

5. Fallout:

    El fallout (también conocido como tasa de falsos positivos) se refiere a la proporción de documentos recuperados que son irrelevantes para la consulta.
    **Interpretación:** Un valor alto de fallout indica que muchos de los documentos recuperados son irrelevantes para la consulta. Es útil cuando se desea evaluar el grado de contaminación en los resultados con documentos irrelevantes.

**Obtención de Evaluaciones para Consultas:** La clase permite obtener las métricas de evaluación para una consulta específica utilizando un modelo de recuperación de información proporcionado. Utiliza los métodos anteriores para calcular las métricas precisión, recall, puntaje F, R-precisión y fallout para la consulta dada.

### Algoritmo de Recomendaciones

Dentro de la clase Recommendation, el método get_recommendations() obtiene 20 o menos recomendaciones basadas en los resultados de una consulta. Se inicia obteniendo todos los documentos de la base de datos a través del objeto de almacenamiento proporcionado durante la inicialización de la clase. Luego, se extraen los títulos y los géneros de estos documentos para su posterior procesamiento.

La vectorización de los títulos se lleva a cabo utilizando TfidfVectorizer, lo que nos permite convertir los títulos en una matriz TF-IDF, una representación numérica que captura la importancia relativa de cada palabra en los títulos.

Una vez que tenemos esta representación numérica de los títulos, calculamos la similitud coseno entre ellos para determinar qué tan similares son entre sí. Esto nos proporciona una medida de similitud que será utilizada más adelante en el proceso de recomendación.

Además, se realizan cálculos sobre los géneros de los documentos. Se cuenta la frecuencia de cada género y se seleccionan los cinco más comunes como géneros principales, que se utilizarán en el proceso de recomendación.

Luego, se procede a calcular las recomendaciones de documentos. Para cada documento que no ha sido recuperado, se calcula la similitud coseno promedio con respecto a los documentos recuperados. Esta similitud se combina con el recuento de género del documento (si pertenece a uno de los géneros principales) para generar una puntuación de recomendación.

Una vez que se han calculado las recomendaciones para todos los documentos, se ordenan según su puntuación y se seleccionan los primeros 20 para ser devueltos como resultado final.

### Insuficiencias

1. **Limitación en la selección de características para la recomendación:**

    En la fase de recomendación, se empleó un conjunto limitado de características. Podríamos haber ampliado este conjunto para incluir más características relevantes, lo que habría mejorado la precisión de las recomendaciones. Además, el método utilizado para definir la distancia entre documentos pudo haber sido más específico. Al utilizar un método más detallado de cálculo de similitud, podríamos haber obtenido recomendaciones más ajustadas a las preferencias del usuario. Además, asumir que todos los documentos tienen título y género puede ser una suposición poco realista, lo que puede afectar la efectividad del sistema de recomendación en ciertos casos.

2. **Falta de una interfaz visual funcional:**

    La ausencia de una interfaz visual funcional dificulta considerablemente probar el código y evaluar su funcionamiento. Aunque se ha realizado un esfuerzo por documentar el código de manera exhaustiva, la falta de una interfaz gráfica funcional puede ser un obstáculo significativo para la eficacia y la usabilidad del sistema. Una interfaz visual proporcionaría una manera más intuitiva de interactuar con el sistema y facilitaría las pruebas y la depuración.

3. **Posible implementación del modelo booleano difuso:**

    Se podría haber implementado el modelo booleano difuso debido a su similitud con los dos modelos propuestos. Esto sugiere una oportunidad perdida para explorar otras alternativas y enriquecer la variedad de modelos utilizados en el proyecto.

4. **Enfoque excesivo en el modelo booleano:**

    Nos enfocamos demasiado en el modelo booleano. Esta falta de atención a otros enfoques y técnicas podría haber limitado la capacidad del sistema para manejar eficazmente diferentes tipos de datos y escenarios. Habría sido beneficioso explorar más a fondo la vectorización y considerar cómo podría integrarse de manera más efectiva con el modelo booleano para mejorar la calidad y la versatilidad de las soluciones propuestas.
