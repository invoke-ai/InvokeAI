import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

import { changelogsLoader } from 'starlight-changelogs/loader';

import { generateDocsId } from './lib/sort-prefix.mjs';

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
    schema: i18nSchema({
      extend: z.object({
        // Replaces `page.editLink` on translated pages, which link to the Crowdin project (`src/route-data.ts`).
        'page.translateLink': z.string().optional(),
      }),
    }),
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
