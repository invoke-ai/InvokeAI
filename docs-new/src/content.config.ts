import { defineCollection } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

import { changelogsLoader } from 'starlight-changelogs/loader';

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
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
      }
    ]),
  })
};
