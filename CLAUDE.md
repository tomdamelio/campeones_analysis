                                                                                                                    
  ## Running Python scripts                                                                                         

  **Never** run scripts with bare `python` or `python3`. Always use:

  micromamba run -n campeones python -m src.campeones_analysis.<module.path>

  For example, to run `src/campeones_analysis/decoding/run_decoding.py`:

  micromamba run -n campeones python -m src.campeones_analysis.decoding.run_decoding

  This applies to every script execution in this project, including one-off runs, tests, and pipeline steps.        

  Una vez guardado, Claude lo cargará automáticamente en esta y todas las sesiones futuras.

  ## Resolver PDFs de papers via Zotero (NO dumpear PDFs al proyecto)

  Para acceder a un paper (en lit-reviews, deep-reads, sub-agentes), **siempre intentar Zotero primero** mediante el helper `zotero_client`. Requiere que Zotero desktop esté corriendo con Local API habilitado (`Edit → Settings → Advanced → "Allow other applications on this computer to communicate with Zotero"`).

  **Resolver un PDF (preferred order: citekey > DOI > PMID > título):**

  ```
  micromamba run -n campeones python -m src.campeones_analysis.lit_review.zotero_client --doi 10.7554/eLife.64812
  ```

  - Exit 0 + stdout = path absoluto al PDF → usar `Read` directo sobre ese path.
  - Exit 1 + stderr informativo = item no encontrado **o** item sin PDF adjunto → pasar al fallback chain.
  - Exit 2 + stderr = error de auth/red (Zotero apagado, BBT roto, etc.).

  Flags soportados: `--doi`, `--pmid`, `--citekey`, `--title`, `--json` (metadata completa en JSON), `--verbose` (debug a stderr).

  **Fallback chain cuando Zotero no tiene el PDF (en este orden):**
  1. PMC OA via `mcp__claude_ai_PubMed__get_full_text_article` (si el paper tiene PMCID).
  2. bioRxiv via `mcp__claude_ai_bioRxiv__get_preprint` (si es preprint).
  3. WebFetch a journal / arXiv.
  4. Abstract only + declarar `Acceso al PDF: abstract only` en el header del notes_<paper>.md.

  **Regla operativa:** NO descargar PDFs al directorio del proyecto (`research_diary/.../pdfs/`). Si un paper relevante NO está en Zotero, decírselo al usuario y proponer que lo agregue (con su PDF) antes de continuar — preferible a duplicar PDFs en el proyecto.

  **Cuando escribís un prompt para un sub-agente** que necesita leer un paper, incluir explícitamente el comando `zotero_client` arriba con el DOI/PMID correspondiente y el fallback chain. Patron template:

  ```
  - **Acceso al PDF:** resolver via Zotero antes de leer. Ejecutar:
    `micromamba run -n campeones python -m src.campeones_analysis.lit_review.zotero_client --doi <DOI>`
    stdout = path absoluto. Si exit != 0: fallback chain (PMC → bioRxiv → WebFetch → abstract only).
  ```

  ## Sincronizar notas profundas a Zotero (al cierre de cada lit-review)

  Tras producir `notes_<paper>.md` con sub-agentes (deep-read), correr el sync para que las notas se attachen como child-notes en los items de Zotero correspondientes:

  ```
  micromamba run -n campeones python -m src.campeones_analysis.lit_review.sync_notes_to_zotero \
      --dir research_diary/context/<MM_NN>/lit_review/ \
      --tag-prefix "CAMPEONES/lit-review/<MM_NN>"
  ```

  Comportamiento:
  - **Idempotente:** re-correr update notes en vez de duplicar (matching via sentinel HTML comment `<!-- claude-sync-notes:<slug> -->`).
  - **Resolución del item parent:** DOI > PMID > title substring > creator+year (todos extraídos del header del notes_*.md).
  - **NO crea items nuevos:** si un paper no está en Zotero, reporta `NO_ITEM` y skipea. El usuario debe agregarlo manualmente (vía "Add Item by Identifier") antes de re-correr.
  - **Tags aplicados:** `<prefix>` al parent + `<prefix>` y `<prefix>/<slug>` a la note.
  - **Markdown → HTML:** vía `markdown` library con extensiones tables/fenced_code/sane_lists.
  - **Flag `--dry-run`** para preview sin escribir.

  Limitaciones conocidas:
  - El Local API tiene flakes ocasionales en `everything()` tras escrituras. Si un paper falla con NO_ITEM y volvés a correr, suele resolverse.
  - Items guardados como `webpage` desde browser frecuentemente no tienen creators/date poblados → falla el fallback creator+year. Solución: editar el item en Zotero desktop y poblar esos campos.

  **Ejemplo de salida típica:**
  ```
    CREATED    notes_hofmann_2021.md  ->  note NF39S75S  on item 57XBSDKK  (by_doi)
    UPDATED    notes_sabbagh_2020.md  ->  note 4RGIA2WQ  on item 6KTAI2PY  (by_doi)
    NO_ITEM    notes_luo_2024_normwear.md  (doi=None pmid=None creator=Luo year=2024)
  ```

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `micromamba run -n campeones graphify query "<question>"` when graphify-out/graph.json exists. Use `micromamba run -n campeones graphify path "<A>" "<B>"` for relationships and `micromamba run -n campeones graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `micromamba run -n campeones graphify update .` to keep the graph current (AST-only, no API cost).
