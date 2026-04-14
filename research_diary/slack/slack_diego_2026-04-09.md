# Mensajes Slack — Diego — 2026-04-09

---

## Mensaje 1: Overview

Hola Diego, te mando un resumen de lo que estuve trabajando esta semana en el proyecto de decoding EEG (luminancia).

La conclusión general es positiva: los datos y el pipeline parecen estar bien. El tope actual de performance parece ser de SNR más que de bugs. Todavía trabajamos con un solo sujeto (sub-27) y 32 canales, lo cual limita bastante. El mejor feature set que tenemos sigue siendo bandpower Welch, con accuracy ~69% y AUC ~0.73 en LORO-CV. No es espectacular, pero es consistente y tiene sentido fisiológico.

---

## Mensaje 2: Sanity check — histogramas alfa/beta

> **Adjuntar:** `results/validation/sanity_checks/sub-27/sub-27_38b_alpha_beta_histograms.png`
> (path absoluto: `C:\Users\au805392\OneDrive - Aarhus universitet\Escritorio\Investigacion\campeones_analysis\results\validation\sanity_checks\sub-27\sub-27_38b_alpha_beta_histograms.png`)

Como sanity check hice histogramas de potencia alfa y beta comparando ventanas pre y post onset del cambio de luminancia en sub-27.

Beta occipital (O1/O2) diferencia claramente (p<0.001): hay beta ERD post-flash, clásico de activación cortical visual. Alpha también diferencia (p=0.029), aunque de forma más modesta. Lo interesante: si el baseline es el inicio del experimento (5 min antes) en lugar de la ventana inmediatamente pre-trial, el alpha diferencia mucho mejor. Eso aparece en el ERP también (siguiente mensaje).

En temporal (T7/T8) no hay diferencias de potencia — la respuesta al tono auditivo sincrónico existe pero es un ERP transitorio, no una modulación sostenida de potencia.

---

## Mensaje 3: ERP + scatter occipital

> **Adjuntar 1:** `results/validation/sanity_checks/sub-27/sub-27_38c_erp_ttest.png`
> (path absoluto: `C:\Users\au805392\OneDrive - Aarhus universitet\Escritorio\Investigacion\campeones_analysis\results\validation\sanity_checks\sub-27\sub-27_38c_erp_ttest.png`)

> **Adjuntar 2:** `results/validation/sanity_checks/sub-27/sub-27_38d_alpha_beta_scatter.png`
> (path absoluto: `C:\Users\au805392\OneDrive - Aarhus universitet\Escritorio\Investigacion\campeones_analysis\results\validation\sanity_checks\sub-27\sub-27_38d_alpha_beta_scatter.png`)

El ERP con t-test FDR confirma respuesta visual en occipital (deflexión ~200–300ms, 30 timepoints significativos) y respuesta auditiva en temporal (~100ms, más temprana que occipital, 39 timepoints significativos).

El scatter alfa×beta en occipital vs temporal (segundo plot) muestra que la separabilidad entre pre y post está casi exclusivamente en O1/O2 — en T7/T8 las nubes se solapan completamente. Eso es exactamente donde esperaríamos el efecto visual, lo cual es tranquilizador.

---

## Mensaje 4: Autocorrelación + conclusión + Yongjie

Adicionalmente probé autocorrelación temporal por canal como feature set alternativo al TDE (siguiendo la pregunta de la semana pasada sobre si la coherencia inter-canal agrega algo). Implementé lags 1–25 (4–100ms a 250Hz, 800 features en total) con LORO-CV y LogisticRegressionCV L2.

Resultado: accuracy ~61.5%, AUC ~0.60 — comparable a TDE con matriz de covarianza (62.2%), pero bastante por debajo de Welch (AUC 0.73). Mi hipótesis es que con ventanas de 500ms la autocorrelación no captura bien las frecuencias bajas — alpha tiene un periodo de 100ms, entonces 500ms son apenas 5 ciclos, y Welch con toda la ventana tiene estimación espectral más robusta. La autocorrelación pierde ahí.

En resumen: mi intuición es que no es un problema del pipeline ni de los datos, sino de SNR. Con un solo sujeto y 32 canales estamos en el límite de lo decodificable con estas features. Extenderlo a más participantes debería mostrar si hay algo más claro ahí.

Por otro lado, tomé contacto con Yongjie Duan (el PhD student con el que mencionaste que estaba trabajando en algo similar). Él usa datos de 128 canales y reporta performances altas. Su approach es bastante diferente al nuestro: aplana el EEG crudo (tiempo × canales) y usa SVM directamente, sin reducción dimensional. Tengo algunas dudas sobre si eso generaliza bien dado el ratio features/trials, pero estamos hablando de probar su pipeline sobre mis datos como chequeo cruzado. ¿Vos tenés algún insight de los resultados que tiene él con eso? Quería ver si hay algo que estemos perdiendo en nuestro approach antes de escalar a más sujetos.

---

## Mensaje 5 — Actualización: probé el pipeline de raw time-series en mis datos

> **Adjuntar:** `results/validation/photo_decoding_mvnn_svc/sub-27_occ_temp_mvnn_interpretability.png`
> (path absoluto: `C:\Users\au805392\OneDrive - Aarhus universitet\Escritorio\Investigacion\campeones_analysis\results\validation\photo_decoding_mvnn_svc\sub-27_occ_temp_mvnn_interpretability.png`)

Bueno, probé el approach de raw time-series (EEG aplanado tiempo x canales + LinearSVC) en este unico sujeto, con el diseño PRE vs POST del mismo evento de flash (ventana 1.2s a 100Hz, Leave one run out CV). Con los 32 canales el accuracy global (promedio de los folds) es 73.0%. La comparación directa con bandpower Welch sobre los mismos datos y el mismo CV: 61.5%. Así que el raw time-series gana por bastante.

Cuando me quede solmanetnte con 8 canales occipitales y temporales (O1, O2, T7, T8, FT9, FT10, TP9, TP10), como regiones de interes para ver cuestiones visuales (flash de luz) y auditivas (tono sonoro que ocurre en simulatneo) el accuracy sube a 77.0%. Los canales frontales y centrales parece que agregan ruido más que señal para esta tarea.

El plot que adjunto es la interpretabilidad: promedio de los pesos del clasificador sobre los 7 folds, reshaped a (8 canales x 120 timepoints). O1 domina claramente, después O2. Así que el modelo esta capturando principalmente la respuesta evocada al onset del flash, segun parece.


Asi que bueno, nos dio insights correr el lo que estbaa haciendo Yongjie, porque de nuevo es ver que son los patrones temporales mas que los de frecuencia, no?