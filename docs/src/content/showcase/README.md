# Showcase entries

Artwork in this directory appears on the docs site's `/showcase/` page, newest first, and the latest eight appear in the
"Made with Invoke" section of the home page. The collection is defined in `docs/src/content.config.ts`.

Winners of the weekly Discord art challenges are published separately, from `src/content/challenges/`: see its
[README](../challenges/README.md).

## Before you add an entry

Add only artwork submitted through the [showcase issue form](https://github.com/invoke-ai/InvokeAI/issues/new?template=SHOWCASE.yml)
with both of its checkboxes ticked: the submitter made the image, and they let Invoke show it on the docs site with
credit. Link that issue in your pull request. If an artist asks for removal, delete both files.

Credit the artist exactly as they wrote it in the form.

## Adding an entry

Each entry is two files with the same name: the image and a YAML file describing it. Name them after the date and the
title, in lowercase with hyphens, for example `2026-11-02-harbour-at-dusk.webp` and `2026-11-02-harbour-at-dusk.yaml`.
The file name, without its extension, becomes the entry's id and the anchor of its tile on `/showcase/`.

```yaml
# 2026-11-02-harbour-at-dusk.yaml
image: ./2026-11-02-harbour-at-dusk.webp
title: Harbour at Dusk
artist: Ana Example # as the artist asked to be credited
artistUrl: https://example.com/ana # optional
model: FLUX.2 Klein # optional, free text
createdWith: Canvas, regional guidance # optional, a short note
date: 2026-11-02 # when the entry was added
```

`image`, `title`, `artist` and `date` are required. `artistUrl` must be an `http://` or `https://` URL. The build fails if a required field is missing or `artistUrl` is anything else.

## Image guidance

- WebP or JPEG. Convert PNG submissions to WebP unless they need lossless detail.
- About 2400 px on the long side. The site generates smaller sizes itself; larger files only slow down the build.
- Keep files under about 1.5 MB.
- Keep the artwork's own aspect ratio and don't crop it. The grid shows every image whole.
