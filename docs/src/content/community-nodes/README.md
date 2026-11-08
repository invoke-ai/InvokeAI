# Community node packs

Each YAML file in this directory is one card on the docs site's Community Nodes page
(`src/content/docs/Users Guide/08.Workflows/community-nodes.mdx`). Cards are sorted by title. The collection is
defined in `docs/src/content.config.ts`, and the build fails when a file breaks its schema.

## Adding a pack

Add one file, named after the pack's title in lowercase with hyphens, such as `curves.yaml`, and optionally its
thumbnail beside it (below). The name, without
`.yaml`, becomes the card's anchor: `/users-guide/workflows/community-nodes/#node-pack-curves`.

```yaml
title: Curves
author: JPPhoto # the repository owner's name
repo: https://github.com/JPPhoto/curves-node # the repository's web page
summary: Adjust an image's curve from a user-defined string. # one sentence, at most 120 characters
categories: [image-effects] # one or more, see below
```

Optional fields:

```yaml
description: >- # more about the pack; separate paragraphs with a blank line
  Works with RGB images only. Wrap node and field names in backticks, as in `collect`; there is no other markup.
install: manual # only when the repository has no top-level __init__.py, so the Nodes manager can't install it
dependencies: true # the pack ships a requirements.txt or pyproject.toml whose packages users install themselves
archived: true # the repository is archived
nodes: # the nodes the pack adds
  - name: Curves
    description: Adjusts the curve. # optional
links: # example workflows, full documentation and other reading
  - label: Example workflow
    url: https://github.com/owner/repo/blob/main/example.json
thumbnail: ./curves.webp # the card's image, beside this file (see below); needs `images`
images: # examples, shown when a reader opens the card's details; the first one's alt text is the thumbnail's too
  - src: https://raw.githubusercontent.com/owner/repo/main/example.png
    alt: What the image shows, for people who can't see it # required
    caption: Output # optional, shown under the image
```

### The thumbnail

The card grid loads only local thumbnails, so it stays fast with dozens of packs; the examples stay in the pack's
repository and load when a reader opens the card's details. Make the thumbnail from your first example: a WebP named
after the YAML file (`curves.webp`), fitting inside 800 × 500 pixels (the card's 16:10 frame at twice its size), under
60 KB. The card shows it whole, letterboxed. For example, with ImageMagick:

```sh
magick example.png -resize '800x500>' -quality 75 curves.webp
```

Lower the quality, or the size, for detailed images that come out larger. The build fails when the file is missing.
Without a thumbnail the card shows its first category's icon.

`categories` takes these ids; the page shows their names in the reader's language:

| Id | Name |
| --- | --- |
| `image-effects` | Image effects |
| `masks` | Masks & backgrounds |
| `compositing` | Compositing & grids |
| `prompts` | Prompt tools |
| `language-models` | Language models |
| `control` | ControlNet & adapters |
| `video-3d` | Video & 3D |
| `sizing` | Sizing & upscaling |
| `utility` | Utility |

A new category goes into `NODE_PACK_CATEGORIES` in `src/lib/community-nodes.mjs`, with its name in
`src/content/i18n/en.json`.

## Before you open the pull request

- The repository is public, and InvokeAI loads it from a clone in its `nodes` folder: an `__init__.py` at the top level
  imports the nodes. Otherwise set `install: manual`.
- `repo`, `links` and image URLs are `https://` and load.
- Example images stay in the pack's repository: use their `raw.githubusercontent.com` (or GitLab raw) URLs.
- The thumbnail is under 60 KB.
- Run `pnpm build` in `docs/` to check the file against the schema, or `pnpm dev` to see the card.

Pack text stays in English, as its author wrote it: Crowdin translates the page around it, not the packs.
