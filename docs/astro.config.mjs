// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
// import rehypeExternalLinks from 'rehype-external-links';

// https://astro.build/config
export default defineConfig({
  markdown: {
    // rehypePlugins: [
    //   [
    //     rehypeExternalLinks,
    //     {
    //       content: { type: 'text', value: ' 🔗' },
    //     },
    //   ],
    // ],
  },

  integrations: [
    starlight({
      // Content
      title: 'Invoke',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/invoke-ai/InvokeAI' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/ZmtBAhwWhy' },
      ],
      sidebar: [
        {
          label: 'Getting Started',
          autogenerate: { directory: 'getting-started' },
        },
        {
          label: 'Configuration',
          autogenerate: { directory: 'configuration' },
        },
      ],
    }),
  ],
});
