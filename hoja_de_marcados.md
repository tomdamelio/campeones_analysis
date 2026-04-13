Por cuestiones de la *toma* (por ejemplo, pausar el bloque y grabar desde 0), hay 3 sesiones que no quedan claro cuando termina un bloque y cuando arranca otro (en terminos de archivos digo):
- 17B
- ⁠20A
- ⁠36B

Cuestiones del *drive*
- Sujeto 32 NO SE TOMÓ TODAVÍA SESIÓN B
- Sujeto 38 NO SE TOMÓ TODAVÍA SESIÓN B
- Sujeto 40 NO SE TOMÓ TODAVÍA SESIÓN B

Los sujetos que no se pueden hacer completos entonces son (desde el 19):
- 20
- 32
- 36
- 38
- 40

Quedan para hacer:


Completos:
- 19 (Un bloque está raro)
- 21 (Comment sobre ses a: SEÑAL DE AUDIO Y FOTO POCO CLARA, LAS MARCAS PARA CORRELACIONAR CON FEATURES DEL VIDEO NO VAN)
- 22 (En el bloque 1a tiene poca variabilidad el reporte, después se normaliza. Seguro no había entendido tan bien el tema del joystick. En el bloque 1b, el orden de los estímulos del xlsx no coincide con el orden en los registros fisiológicos, aunque si los videos. Habría que ver el video desde la carpeta de Gizmo, para ver cuál de los dos realmente corresponde a lo que vio el sujeto [podría ser que el registro sea de otro sujeto, y ese problema no este solucionado al final, o podría ser que el video corresponda al registro, y entonces probablemente el video no era el que había que pasarle para que vea]) (EDA VISTO, SE VE MEJOR EN LA SESION B QUE EN LA A)
- 23 (EDA VISTO, UN BLOQUE RARO)
- 24
- 25 (Sesión A impecable. Sesión B imposible ver los marcadores de foto y audio, demasiado ruido :/ )
- 26 (Sesión A bloque 2 no está completo, habría que rechequear con los archivos originales a ver que onda. En las anotaciones de la sesión no hay nada, y los otros bloques están todos ok.)
- 27
- 28 (EDA VISTO, VARIOS BLOQUES DEFECTUOSOS O INCLUSO PERDIDOS)
- 29 (Sesión A impecable. Sesión B imposible ver los marcadores de foto y audio, demasiado ruido :/ )
- 30 (EDA VISTO, ESTÁ BIEN PERO PLANCHADO)
- 31 (En el bloque 2 de la sesión B tuve que dejar el primer estimulo más largo, porque extrañamente duraba menos [aproximadamente 3 segundos] que lo que decía el xlsx de orden. Después todo excelente)
- 33
- 34 (Sesión A es imposible ver los marcadores. Algo se nota en el canal de audio [dos picos para ambos lados del espectro], pero no sé donde poner la marca exactamente. Sesión B está bien todo menos el bloque 2, que dura muy poco. También pareciera incluso que las marcas no están sincronizadas con el reporte? Como que hay partes donde el reporte sigue anuque haya habido una marca. No se entiende bien)
- 35 (Sesión A es imposible ver los marcadores. Sesión B está todo bien.)
- 37 (Sesión A es imposible ver los marcadores.) (EDA VISTO, HAY UN BLOQUE MEDIO RUIDOSO DE MÁS, IGUALMENTE EL BLOQUE 2 DE LA SESIÓN A SE VE MUY CORTO)
- 39 (Sesión A bloque 1, bloque 3 y bloque 4 algunos videos duraban apenas un poquito menos de lo que deberían [el onset + la duración caía en el medio de la señal que representa el marcador]. Sesión B no hay reporte con el joystick, así que al pedo hacer las marcas creo) (EDA VISTO, HAY UN BLOQUE RUIDOSO PERO QUIZAS SALVABLE)

Proceso:
1. python -m src.campeones_analysis.physio.read_xdf --subject XX

2. python scripts/preprocessing/02_create_events_tsv.py --subjects XX --all-runs

3. python scripts\sanity_check\physio_validation_check.py
- Si la señal de EDA está ok, seguir. Si no, no perder tiempo

4. python scripts/preprocessing/03_detect_markers.py --subject XX --task YY --acq Z

Otros pensamientos a partir de este pre-procesamiento:
- Para extracción de features, si lo vamos a hacer a nivel bloque y no a nivel estímulo, habría que agarrar unos segundos antes del primer evento, y unos segundos después, porque antes y después de eso puede haber MUCHO ruido.