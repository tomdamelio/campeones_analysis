---
inclusion: fileMatch
fileMatchPattern: "**/*.md"
---

# Protocolo para editar archivos Markdown con caracteres no-ASCII

## Problema

Los archivos `.md` de este proyecto (research diary, READMEs) contienen español
con caracteres UTF-8 multi-byte (ñ, ó, á, é, ü, ×, etc.). Las herramientas
`strReplace` y `editCode` fallan silenciosamente cuando el `oldStr` no matchea
byte-a-byte, lo cual ocurre frecuentemente con estos caracteres.

## Regla: NO usar strReplace/editCode en .md con caracteres no-ASCII

Cuando necesites editar un archivo Markdown que contenga caracteres no-ASCII:

1. **NUNCA** uses `strReplace` o `editCode` con `old_str`/`new_str` que contenga
   acentos, ñ, ×, ±, o cualquier caracter no-ASCII. Va a fallar y vas a entrar en loop.

2. **USA `fsWrite` directamente** con este flujo:
   - Leer el archivo con `readFile` para ver el contenido actual y los numeros de linea.
   - Leer las partes que NO cambian con `readFile` por rangos (start_line/end_line).
   - Construir el contenido nuevo combinando las partes sin cambios + el texto nuevo.
   - Escribir con `fsWrite` el archivo completo (o `fsWrite` + `fsAppend` si es largo).
   - **NO crear scripts Python temporales.** Editar directamente.

3. **Ejemplo de flujo:**
   - Quiero reemplazar lineas 50-60 del diary.
   - `readFile` lineas 1-49 -> guardo mentalmente como "parte_antes"
   - `readFile` lineas 61-final -> guardo mentalmente como "parte_despues"
   - `fsWrite` con: parte_antes + contenido_nuevo + parte_despues

4. **Para agregar contenido al final:** usar `fsAppend` directamente.

## Para ediciones simples sin caracteres especiales

Si el `oldStr` es puramente ASCII (solo letras inglesas, numeros, simbolos basicos),
`strReplace` funciona bien. Usalo normalmente en ese caso.

## Resumen de decision

- El `oldStr` o `newStr` tiene ñ, á, é, í, ó, ú, ü, ×, o cualquier caracter no-ASCII?
  -> `fsWrite` directo (leer por rangos + reescribir)
- Es todo ASCII puro?
  -> `strReplace` / `editCode` esta bien
