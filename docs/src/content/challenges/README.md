# Weekly challenge winners

Each folder in this directory is one week of the Invoke Discord's art challenge. The latest week appears at the top of
the docs site's `/showcase/` page and in the "Made with Invoke" section of the home page; every week has its own page
at `/showcase/challenges/<folder name>/` and a card on `/showcase/challenges/`. Weeks are ordered by `endDate`, newest
first. The collection is defined in `docs/src/content.config.ts`.

`startDate` is the day the challenge was announced and `endDate` the day its winners were announced. The next
challenge is usually announced with the winners, so one week's `endDate` is often the next week's `startDate`; that is
expected. `discordUrl` links the winners' announcement, shown as "Winners on Discord".

## Consent and credit

Entrants agree, by entering a challenge on Discord, that their entry may be featured on the docs site. Publish only
entries from the challenge itself, and credit each winner by name exactly as they go by on Discord, or as they ask to be
credited. Add a link only if they gave you one.

**Removing an entry on request.** If a winner asks to be removed, delete their item from `winners` in
`challenge.yaml` and delete their image file, then open a pull request. Leave the other winners' places as they are:
the page leads with the first winner left, which is the highest place. A week needs a numbered winner, so if they held
the only numbered place, or were the week's only winner, delete the whole folder. If they ask only for a different
credit or no link, edit `artist` or remove `artistUrl`.

## Publishing a week

1. **Make the folder.** Name it `YYYY-MM-DD-<theme>`, with `startDate` and the theme in lowercase letters, digits and
   hyphens, such as `2026-10-19-biomechanical-pets`. Only if Discord announced a week number, name it
   `<year>-w<week>-<theme>` instead, with the year of `startDate` and that number, such as
   `2026-w44-bioluminescence`. The name becomes the page's address, so don't rename a folder once it is published;
   links shared on Discord would break.
2. **Add the images** to the folder (see [Exporting images](#exporting-images)). Name them after the place and the
   artist, such as `1-ana-example.webp`, `2-mira-k.webp`, `runner-up-theo.webp`, `honourable-lumen.webp`.
3. **Write `challenge.yaml`** in the folder, using the example below.
4. **Preview it** (see [Previewing locally](#previewing-locally)), then open a pull request with the folder.

Folder layout:

```text
src/content/challenges/
└── 2026-10-21-bioluminescence/
    ├── challenge.yaml
    ├── 1-ana-example.webp
    ├── 2-mira-k.webp
    ├── runner-up-theo.webp
    └── honourable-lumen.webp
```

### `challenge.yaml`

Every field, with the optional ones marked:

```yaml
theme: Bioluminescence # as announced
brief: Living light in the deep, the forest floor or the city at night. # optional, as announced, at most 280 characters
startDate: 2026-10-21 # YYYY-MM-DD, without quotes: the day the challenge was announced
endDate: 2026-10-27 # YYYY-MM-DD, without quotes: the day the winners were announced
discordUrl: https://discord.com/channels/1020123559063990373/1020355799945580585/1564197824751734795 # optional
winners: # in the order the page shows them
  - place: 1 # 1, 2, 3, runner-up or honourable
    image: ./1-ana-example.webp # the file in this folder
    artist: Ana Example # the credit, exactly as given
    artistUrl: https://example.com/ana # optional, http:// or https:// only
    title: Tidewater Lantern # optional
    model: FLUX.2 Klein # optional, free text
  - place: 2
    image: ./2-mira-k.webp
    artist: mira_k
  - place: runner-up # any number of runners-up, shown in this order
    image: ./runner-up-theo.webp
    artist: Theo
  - place: honourable # any number of honourable mentions, shown in this order
    image: ./honourable-lumen.webp
    artist: lumen
```

A numbered week (folder `2026-w44-bioluminescence`) adds the announced number as the first line: `week: 44`.

Rules the build and tests enforce:

- `theme`, `startDate`, `endDate` and `winners` are required, and so are each winner's `place`, `image` and
  `artist`. A misspelt field name, such as `artistURL`, fails the build rather than being ignored.
- Winners are listed in the order the page shows them: numbered places first, never decreasing (`1, 2, 2` for a tie
  for 2nd; `2, 1` fails), then runners-up, then honourable mentions. At most three winners hold a numbered place,
  ties included, and at least one does: the first winner leads the page and is the preview image when the page is
  shared on Discord.
- Dates are written `YYYY-MM-DD` without quotes: `"2026-10-21"` in quotes is text, not a date, and fails the build.
  `endDate` is on or after `startDate`.
- A dated folder is named `YYYY-MM-DD-<theme>` with its `startDate`, and has no `week`. A numbered folder is named
  `<year>-w<week>-<theme>`, with the year `startDate` falls in and the same number as `week`. A week number is used
  once per year, so the folders work whether Discord's numbering restarts each year or not, and a start date once
  across all folders.
- The folder holds only `challenge.yaml` and the images it names: no leftover or unused files.
- Images are WebP or JPEG, each under 500 KB (500,000 bytes).

Guidance the checks can't enforce:

- Use the places, theme and brief as announced on Discord, and copy the announced text as written. Use `runner-up`
  for winners announced as runners-up without a place, and `honourable` for honourable mentions.
- Never invent a week number, from a calendar or by counting folders: leave `week` out unless Discord announced one.

An image without a `title` is described for screen readers by the theme and the place, such as "Bioluminescence, 1st
place by Ana Example" or "Lost, runner-up by Romeo"; with a title, as "Tidewater Lantern by Ana Example".

## Exporting images

- **WebP** (or JPEG), **at most 2400 pixels on the long side** (ideally 1600 or more; 2000 to 2400 is best), **under
  500 KB**. The site makes smaller sizes itself.
- Start from the original file the entrant posted. Never save Discord's preview of it: previews are scaled down, to
  718 pixels tall or less.
- Never upscale: export a smaller entry at its own size. Enlarging adds blur and bytes, not detail.
- Keep the artwork's own aspect ratio and don't crop it: the winners' layout shows every image whole.
- Convert PNG entries, without transparency; don't commit the PNG.

With ImageMagick (`magick`, or `convert` on older versions). The `>` in `'2400x2400>'` only ever shrinks, so smaller
entries keep their size:

```sh
magick entry.png -resize '2400x2400>' -quality 80 1-ana-example.webp
```

Or with `cwebp`, which has no shrink-only option. If the long side is over 2400 pixels, resize: `-resize 2400 0` sets
the width (for a portrait image, `-resize 0 2400` sets the height):

```sh
cwebp -q 80 -resize 2400 0 entry.png -o 1-ana-example.webp
```

If the long side is 2400 pixels or less, leave out `-resize`:

```sh
cwebp -q 80 entry.png -o 1-ana-example.webp
```

Check the size: `ls -l` shows bytes. If it is over, lower the quality to 70, or the long side to 2000 pixels, and
export again.

## Previewing locally

From the repository's `docs/` directory, with Node 22.23.4 or newer and pnpm 10 (see `docs/README.md`):

```sh
pnpm install
pnpm dev
```

Open `http://localhost:4321/showcase/` and `http://localhost:4321/showcase/challenges/<folder name>/`. A mistake in
`challenge.yaml` shows as an error in the terminal and the browser. Then run the checks the pull request runs:

```sh
pnpm test   # folder contents and names, image formats and sizes, unique weeks and start dates
pnpm build  # the schema: fields, dates and the winners' places and order
```

Challenge text stays in English, as written: the showcase pages are not translated.
