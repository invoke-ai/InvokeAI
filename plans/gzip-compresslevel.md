# GZip: `compresslevel` konfigurierbar bzw. abschaltbar machen

> **Status:** umgesetzt auf `fix/routes-block-event-loop` — `gzip_compresslevel` (Default 1,
> 0 = aus) in `InvokeAIAppConfig`, Einhängen über `configure_gzip()` in `api_app.py`, Tests in
> `tests/app/api/test_gzip_content_types.py`, Doku-Abschnitt „Response Compression" in
> `configuration/invokeai-yaml.mdx`. Vorgelagert erledigt: Medien-Content-Types werden nicht
> mehr komprimiert (`ContentTypeAwareGZipMiddleware`).
> **Alle Zahlen unten sind gemessen**, nicht geschätzt — Messskripte im Scratchpad
> (`gzip_cost.py`, `gzip_levels.py`).

## Warum

`app.add_middleware(ContentTypeAwareGZipMiddleware, minimum_size=1000)` in
`invokeai/app/api_app.py` übergibt kein `compresslevel`. Starlettes Default ist **9** — die
langsamste Stufe. Kompression läuft vollständig auf dem Event-Loop, blockiert also für ihre
Dauer den gesamten Prozess (keine anderen Requests, keine socket.io-Events).

Gemessen an der flachen Namensliste einer 200k-Bibliothek (8,48 MB JSON, produktive
UUID-Dateinamen):

| Level | Zeit | Ergebnis | Anteil |
| --- | ---: | ---: | ---: |
| 1 | **16,4 ms** | 0,52 MB | 6,1 % |
| 3 | 16,6 ms | 0,52 MB | 6,1 % |
| 6 | 36,1 ms | 0,50 MB | 5,9 % |
| 9 (Default) | **90,2 ms** | 0,48 MB | 5,7 % |

Level 9 kostet das **5,5-fache** an CPU für **0,4 Prozentpunkte** kleinere Ausgabe. Bei
lokalem Betrieb — dem Normalfall — ist die eingesparte Bandbreite wertlos, die Loop-Blockade
dagegen direkt spürbar.

Das deckt sich mit der Messung am neuen `item_names`-Endpunkt: dessen Rest-Blockade von
~102 ms p95 besteht praktisch vollständig aus GZip.

## Was zu tun ist

1. **Config-Option** in `InvokeAIAppConfig` (`invokeai/app/services/config/config_default.py`),
   z. B. `gzip_compresslevel: int = Field(default=1, ge=0, le=9)`, wobei **0 = aus** bedeutet.
   Bei 0 die Middleware gar nicht erst einhängen, statt mit Level 0 zu komprimieren — sonst
   zahlt man weiterhin den Durchlauf durch den Responder.
2. **Default auf 1 senken.** Die Tabelle oben ist die Begründung; 0,4 Prozentpunkte sind kein
   vertretbarer Preis für 74 ms Loop-Blockade pro Namensanfrage.
3. **Reverse-Proxy-Fall dokumentieren.** Wer hinter nginx/Caddy deployt, lässt dort
   komprimieren und will `gzip_compresslevel=0` — dann wird zweifache Kompression vermieden.

## Was ausdrücklich *nicht* die Lösung ist

Den Level zu senken hilft **nicht** gegen den Medien-Fall. Auf inkompressiblen Daten
(PNG/WebP/MP4 sind bereits deflate-komprimiert) kostet Level 1 praktisch dasselbe wie Level 9,
weil Deflate die Daten trotzdem vollständig scannen muss:

| Nutzlast | Level 9 | Level 1 | Ergebnis |
| --- | ---: | ---: | --- |
| 1024×1024 PNG (3,00 MB) | 52,2 ms | 51,0 ms | 3,01 MB — **größer** als das Original |
| 2048×2048 PNG (12,02 MB) | 209,7 ms | 201,7 ms | 12,02 MB — keine Ersparnis |

Deshalb ist der Content-Type-Ausschluss die eigentliche Behebung dieses Falls und bereits
umgesetzt; die Config-Option adressiert den *komprimierbaren* Pfad.

## Testbarkeit

`tests/app/api/test_gzip_content_types.py` deckt bereits ab, *welche* Typen komprimiert werden.
Für diese Aufgabe kommt hinzu: bei `gzip_compresslevel=0` darf keine Antwort ein
`Content-Encoding: gzip` tragen.

## Verwandt

- `docs/src/content/docs/contributing/blocking-work-in-api-routes.md` — warum Arbeit auf dem
  Event-Loop den ganzen Prozess anhält.
- Offen aus derselben Untersuchung: **(c)** Cache-Invalidierung pro fertigem Bild in
  `onInvocationComplete.tsx` (löst die teure Namensabfrage 20-mal pro Batch aus) und
  **(D)** der FastAPI-Pin auf 0.118.3 wegen des `AnyInvocation`-OpenAPI-Bugs.
