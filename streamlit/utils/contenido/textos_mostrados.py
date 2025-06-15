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
A partir de la matriz de confusión podemos decir que, si bien presenta una cantidad significativa de falsos positivos, esto es una decisión consciente, ya que el objetivo del proyecto (tal como ya habiamos mencionado anteriormente) es maximizar la retención de los clientes dado que el costo de  intervenir sobre uno que no se iba a dar de baja es menor que el costo de perder uno que sí. Es por esto que sacrificamos la precisión del modelo pero con la finalidad de detectar mas churn.

El modelo final basado en SVM logró una balanced accuracy de 75%, lo cual indica un buen desempeño general incluso frente al desbalance de clases. La métrica más relevante para nuestro objetivo, recall para la clase "churn", alcanzó un valor de 0.78, lo que implica que el modelo es capaz de identificar correctamente a casi 8 de cada 10 clientes que efectivamente se darían de baja.

Si bien la precision es de 0.50, esto significa que la mitad de las predicciones positivas son aciertos, lo cual consideramos aceptable dado que el costo de una intervención preventiva mal dirigida (falso positivo) es considerablemente menor al costo de perder un cliente (falso negativo).

En este sentido, el modelo cumple con el objetivo principal del negocio: maximizar la capacidad de detectar posibles bajas para poder intervenir a tiempo. El balance entre recall y precision, reflejado en un F1 Score de 0.61, refuerza esta idea.

Finalmente, la métrica PR AUC (área bajo la curva de precisión-recall) también confirma que el modelo tiene una capacidad de discriminación adecuada en un entorno con clases desbalanceadas.


""".strip()

texto_variables_mas_importantes = """
1. `tenure`: La antigüedad de un cliente es el predictor mas importante con diferencia. Esto era algo esperable, ya que suena lógico pensar que aquellos clientes con mayor antigüedad son los mas fieles y, por ende, los menos propensos a churnear.
2. `Contract_One year`y `Contract_Two year`: Los tipos de contrato también tienen un peso considerable en el modelo. Ya habíamos observado en el EDA que los clientes con contratos de dos años, en su mayoría, no realizaron churn. Esto sugiere que los compromisos contractuales más largos funcionan como mecanismos de retención efectivos.

3. `MonthlyCharges`: Es posible que los cargos más elevados estén asociados a una mayor probabilidad de abandono, ya sea por percepción de alto costo o falta de relación precio-calidad.

4. `InternetService_No`: No contar con servicio de internet es un factor que afecta la permanencia. Muchas de las variables categóricas relacionadas con servicios adicionales, como HasStreamingTV o HasOnlineSecurity, dependen de tener conexión a internet. Por lo tanto, los clientes sin este servicio suelen tener una menor cantidad de productos contratados, lo que puede traducirse en un menor nivel de compromiso con la empresa y, en consecuencia, una mayor probabilidad de churn.

5. `PaymentMethod_Electronic check`: Tal como se observó en el EDA, los clientes que utilizan el pago mediante cheque electrónico presentan una mayor tasa de churn. Esto puede deberse a que este método de pago está asociado a clientes menos digitalizados o con menor fidelización, lo que el modelo logró capturar como un patrón relevante a comparación de las demas variables.

""".strip()