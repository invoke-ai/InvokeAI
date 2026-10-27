import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

import { changelogsLoader } from 'starlight-changelogs/loader';

import en from './content/i18n/en.json';
import { NODE_PACK_CATEGORIES } from './lib/community-nodes.mjs';
import { generateDocsId } from './lib/sort-prefix.mjs';

// `en.json` declares the site's own UI strings (and the few Starlight strings it rewords), which Crowdin translates
// into each locale's file. Its keys are the schema, so a string is declared once.
const uiStrings = z.object(
  Object.fromEntries(Object.keys(en).map((key) => [key, z.string().optional()])) as {
    [Key in keyof typeof en]: z.ZodOptional<z.ZodString>;
  }
);

export const collections = {
  docs: defineCollection({
    loader: docsLoader({ generateId: generateDocsId }),
    schema: docsSchema({
      extend: z.object({
        // Docs directories, as URL paths (e.g. `users-guide/models/local-models`), whose sidebar groups also list
        // this page. See `src/route-data.ts`.
        alsoListedIn: z.array(z.string()).optional(),
      }),
    }),
  }),
  i18n: defineCollection({
    loader: i18nLoader(),
    schema: i18nSchema({ extend: uiStrings }),
  }),
  // Community artwork for `/showcase/` and the home page. `src/content/showcase/README.md` explains how to add one.
  showcase: defineCollection({
    loader: glob({ pattern: '**/*.{yaml,yml,json}', base: './src/content/showcase' }),
    schema: ({ image }) =>
      z.object({
        image: image(),
        title: z.string(),
        /** The artist's name as they asked to be credited. */
        artist: z.string(),
        /** Only http(s): it becomes a link, and its value comes from a public submission form. */
        artistUrl: z.httpUrl().optional(),
        /** Free text, such as "FLUX.2 Klein". */
        model: z.string().optional(),
        /** A short note on how it was made, such as "Canvas, regional guidance". */
        createdWith: z.string().optional(),
        date: z.coerce.date(),
      }),
  }),
  // Node packs on the Community Nodes page. `src/content/community-nodes/README.md` explains how to add one.
  communityNodes: defineCollection({
    loader: glob({ pattern: '*.yaml', base: './src/content/community-nodes' }),
    // Strict objects, so a misspelt field (`dependency`, `thumbnial`) fails the build instead of being dropped.
    schema: ({ image }) =>
      z
        .strictObject({
          title: z.string(),
          /** The repository owner's name, as on GitHub or GitLab. */
          author: z.string(),
          /** The repository's web page. With `.git` appended it is the URL the Nodes manager and `git clone` take. */
          repo: z.httpUrl(),
          /** One sentence for the card. */
          summary: z.string().max(120),
          /** More about the pack: paragraphs separated by a blank line. `code` spans are the only markup. */
          description: z.string().optional(),
          categories: z
            .array(z.enum(NODE_PACK_CATEGORIES.map(({ id }) => id) as [string, ...string[]]))
            .min(1),
          /** `manual` when the repository is not a pack folder (no top-level `__init__.py`), so the manager can't install it. */
          install: z.enum(['manager', 'manual']).default('manager'),
          /** The pack ships a `requirements.txt` or `pyproject.toml`, whose packages you install yourself. */
          dependencies: z.boolean().default(false),
          /** The repository is archived: read-only, with no further updates. */
          archived: z.boolean().default(false),
          nodes: z.array(z.strictObject({ name: z.string(), description: z.string().optional() })).default([]),
          /** Further reading, such as example workflows. */
          links: z.array(z.strictObject({ label: z.string(), url: z.httpUrl() })).default([]),
          /**
           * The card's image: a small WebP of the first example beside this file (see the README), so the grid loads no
           * remote images. The build fails when the file is missing. Its alt text is the first example's.
           */
          thumbnail: image().optional(),
          /** Hosted by the pack's author and loaded only when a reader opens the card's details. */
          images: z
            .array(z.strictObject({ src: z.httpUrl(), alt: z.string().min(1), caption: z.string().optional() }))
            .default([]),
        })
        .refine((pack) => !pack.thumbnail || pack.images.length > 0, {
          message: 'A thumbnail takes its alt text from the first image, so `images` needs at least one entry.',
          path: ['thumbnail'],
        }),
  }),
  changelogs: defineCollection({
    loader: changelogsLoader([
      {
        title: "Releases",
        provider: 'github',
        base: 'releases',
        owner: 'invoke-ai',
        repo: 'InvokeAI',
        pagefind: false,
        // Authenticate GitHub API requests so the release changelog loader uses
        // the 5000 req/hr authenticated rate limit instead of the 60 req/hr
        // unauthenticated limit (shared per CI runner IP), which causes
        // intermittent "403 - rate limit exceeded" build failures. The token is
        // optional, so local builds without it fall back to unauthenticated.
        token: process.env.GITHUB_TOKEN,
      }
    ]),
  })
};
