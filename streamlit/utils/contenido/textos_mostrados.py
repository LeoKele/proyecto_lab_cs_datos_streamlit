texto_introduccion = """
En el presente trabajo desarrollaremos un proyecto completo de análisis y modelado predictivo sobre el abandono de clientes (“churn”) en una compañía de telecomunicaciones. Dado que adquirir un nuevo cliente puede ser bastante más costoso que retener a uno existente, contar con una predicción fiable de churn se traduce en un importante beneficio tanto económico como de satisfacción para la compañía.


El dataset consta de 7 043 observaciones y 21 variables que describen información demográfica (género, edad, estado de pareja, dependientes), datos de la cuenta (tiempo de permanencia, tipo de contrato, método de pago, facturación sin papel), cargos monetarios (cargos mensuales y totales) y servicios contratados (telefonía, múltiples líneas, Internet y servicios asociados como seguridad en línea, copias de respaldo, soporte técnico o streaming).

La variable respuesta `Churn` indica si el cliente canceló su contrato en el último mes. Siguiendo la metodología del ciclo de vida de un proyecto propuesta en la asignatura, incluiremos identificación del problema, limpieza de datos, exploración y visualización, selección y ajuste del modelo y finalmente una evaluación del modelo.
""".strip()



texto_problema = """
Reconocemos que reducir el churn es crítico: captar un nuevo cliente [puede costar de cinco a veinticinco veces más que retener uno existente](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers). Por ello, nuestro objetivo es predecir con antelación qué usuarios tienen mayor probabilidad de cancelar el servicio.

Un modelo predictivo de churn permite identificar a aquellos clientes con alta probabilidad de abandono y aplicarles de manera selectiva acciones de retención (ofertas personalizadas, upgrades de servicio, atención proactiva), cuya inversión es considerablemente más baja que la necesaria para reemplazarlos por nuevos usuarios. Esto no solo incrementa la eficiencia del presupuesto de marketing, sino que fortalece las métricas de satisfacción y de lealtad, alimentando un círculo virtuoso de crecimiento sostenible.

""".strip()




texto_conclusion_modelo_final = """
A partir de la matriz de confusión, podemos observar que, si bien el modelo presenta una cantidad significativa de falsos positivos, esto se debe a una decisión consciente. Tal como mencionamos anteriormente, el objetivo del proyecto es maximizar la retención de clientes, dado que el costo de intervenir sobre un cliente que no se iba a dar de baja es menor que el costo de perder a uno que sí. Por esta razón, estamos dispuestos a sacrificar parte de la precisión del modelo con el fin de detectar una mayor cantidad de casos de churn.

El modelo final, basado en support vector machines (SVM), alcanzó una balanced accuracy del 75 %, lo que indica un buen desempeño general incluso frente al desbalance de clases. La métrica más relevante para nuestro objetivo, el recall para la clase "churn", obtuvo un valor de 0.78, lo que significa que el modelo es capaz de identificar correctamente a casi 8 de cada 10 clientes que efectivamente se darían de baja.

Si bien la precisión fue de 0.50, lo que implica que la mitad de las predicciones positivas fueron aciertos, consideramos este resultado aceptable, dado que el costo de una intervención preventiva mal dirigida (falso positivo) es considerablemente menor al de perder un cliente que efectivamente abandona (falso negativo).

En este sentido, el modelo cumple con el objetivo principal del negocio: maximizar la capacidad de detección de posibles bajas para intervenir a tiempo. El equilibrio alcanzado entre recall y precisión, reflejado en un F1 Score de 0.61, respalda esta decisión.

Por último, la métrica PR AUC (área bajo la curva precisión-recall) también confirma que el modelo tiene una capacidad de discriminación adecuada en un entorno con clases desbalanceadas.


""".strip()

texto_variables_mas_importantes = """
1. `tenure`: La antigüedad del cliente es, por amplio margen, el predictor más importante del modelo. Este resultado era esperable, ya que resulta lógico pensar que los clientes con mayor antigüedad tienden a ser más fieles y, por lo tanto, menos propensos a abandonar el servicio. Esta relación ya había sido identificada durante el análisis exploratorio, al comparar los boxplots de ambas clases dentro de la variable tenure.

2. `Contract_One year`y `Contract_Two year`: Los tipos de contrato también tienen un peso considerable en el modelo. En el EDA habíamos observado que los clientes con contratos de dos años, en su mayoría, no realizaban churn. Esto sugiere que los compromisos contractuales más prolongados actúan como mecanismos efectivos de retención.

3. `MonthlyCharges`: Es posible que cargos mensuales más elevados estén asociados a una mayor probabilidad de abandono, ya sea por la percepción de un costo excesivo o por una falta de relación entre precio y calidad del servicio recibido.

4. `InternetService_No`: No contar con servicio de internet se presenta como un factor que influye en la permanencia del cliente. Muchas de las variables categóricas relacionadas con servicios adicionales, como HasStreamingTV o HasOnlineSecurity, dependen de la existencia de una conexión a internet. Por lo tanto, los clientes que no cuentan con este servicio tienden a contratar menos productos, lo que se traduce en un menor nivel de compromiso con la empresa y, en consecuencia, en una mayor probabilidad de churn.

5. `PaymentMethod_Electronic check`: Tal como se observó en el EDA, los clientes que utilizan el pago mediante cheque electrónico presentan una mayor tasa de abandono. Esto podría deberse a que este método de pago está asociado a clientes menos digitalizados o con menor fidelización, lo cual el modelo logró capturar como un patrón relevante en comparación con otras variables.

""".strip()