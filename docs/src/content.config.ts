import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { z } from 'astro/zod';

import { changelogsLoader } from 'starlight-changelogs/loader';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  i18n: defineCollection({
    loader: i18nLoader(),
    schema: i18nSchema({
      extend: z
        .object({
          'footer.builtBy': z.string(),
          'page.translateLink': z.string(),
          'download.windows.headline': z.string(),
          'download.windows.note': z.string(),
          'download.windows.action': z.string(),
          'download.macos.headline': z.string(),
          'download.macos.note': z.string(),
          'download.macos.action': z.string(),
          'download.linux.headline': z.string(),
          'download.linux.note': z.string(),
          'download.linux.action': z.string(),
          'download.github.headline': z.string(),
          'download.github.description': z.string(),
          'download.docker.headline': z.string(),
          'download.docker.description': z.string(),
          'download.hosted.heading': z.string(),
          'download.hosted.description': z.string(),
          'download.aibadgr.headline': z.string(),
          'download.aibadgr.description': z.string(),
          'download.runpod.headline': z.string(),
          'download.runpod.description': z.string(),
          'download.railway.headline': z.string(),
          'download.railway.description': z.string(),
          'download.separator': z.string(),
          'systemRequirements.title': z.string(),
          'systemRequirements.description': z.string(),
          'settings.type': z.string(),
          'settings.default': z.string(),
          'settings.environment': z.string(),
          'settings.values': z.string(),
        })
        .partial(),
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
