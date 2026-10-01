import { defineCollection } from 'astro:content';
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
  i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
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
