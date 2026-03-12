Por cuestiones de la *toma* (por ejemplo, pausar el bloque y grabar desde 0), hay 3 sesiones que no quedan claro cuando termina un bloque y cuando arranca otro (en terminos de archivos digo):
- 17B
- ⁠20A
- ⁠36B

Cuestiones del *drive*
- Sujeto 22 falta excel de orden de sesión B bloque 3
- Sujeto 24 faltan excels de orden de sesión B (y datos también)
- Sujeto 32 NO SE TOMÓ TODAVÍA SESIÓN B
- Sujeto 38 NO SE TOMÓ TODAVÍA SESIÓN B
- Sujeto 40 NO SE TOMÓ TODAVÍA SESIÓN B

Los sujetos que no se pueden hacer completos entonces son (desde el 19):
- 20
- 22
- 24
- 32
- 36
- 38
- 40

Quedan para hacer:
- 19 EDA VISTO, UN BLOQUE RARO
- 23 EDA VISTO, UN BLOQUE RARO
- 28 EDA VISTO, VARIOS BLOQUES DEFECTUOSOS O INCLUSO PERDIDOS
- 30
- 31
- 33
- 34
- 35
- 37
- 39 

Completos:
- 21 (Comment sobre ses a: SEÑAL DE AUDIO Y FOTO POCO CLARA, LAS MARCAS PARA CORRELACIONAR CON FEATURES DEL VIDEO NO VAN)
- 25 (Sesión A impecable. Sesión B imposible ver los marcadores de foto y audio, demasiado ruido :/ )
- 26 (Sesión A bloque 2 no está completo, habría que rechequear con los archivos originales a ver que onda. En las anotaciones de la sesión no hay nada, y los otros bloques están todos ok.)
- 27
- 29 (Sesión A impecable. Sesión B imposible ver los marcadores de foto y audio, demasiado ruido :/ )

Proceso:
1. python -m src.campeones_analysis.physio.read_xdf --subject XX

2. python scripts/preprocessing/02_create_events_tsv.py --subjects XX --all-runs

3. python scripts\sanity_check\physio_validation_check.py
- Si la señal de EDA está ok, seguir. Si no, no perder tiempo

4. python scripts/preprocessing/03_detect_markers.py --subject XX --task YY --acq Z

Otros pensamientos a partir de este pre-procesamiento:
- Para extracción de features, si lo vamos a hacer a nivel bloque y no a nivel estímulo, habría que agarrar unos segundos antes del primer evento, y unos segundos después, porque antes y después de eso puede haber MUCHO ruido.