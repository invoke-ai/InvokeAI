// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
import markdoc from '@astrojs/markdoc';
// import { strategy } from 'sharp';
// import rehypeExternalLinks from 'rehype-external-links';

// https://astro.build/config
export default defineConfig({
  // markdown: {},

  integrations: [
    starlight({
      // Content
      title: 'Invoke',
      logo: {
        src: './src/assets/invoke-icon-wide.svg',
        alt: 'InvokeAI Logo',
        replacesTitle: true,
      },
      favicon: './src/assets/invoke-icon.svg',
      editLink: {
        baseUrl: 'https://github.com/invoke-ai/InvokeAI/tree/main/docs',
      },
      // locales: {
      //   en: {
      //     label: 'English',
      //   },
      // },
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/invoke-ai/InvokeAI' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/ZmtBAhwWhy' },
      ],
      tableOfContents: {
        maxHeadingLevel: 4,
      },
      sidebar: [
        {
          label: 'Getting Started',
          autogenerate: { directory: 'getting-started' },
        },
        {
          label: 'Configuration',
          autogenerate: { directory: 'configuration' },
        },
        {
          label: 'Development',
          autogenerate: { directory: 'development' },
        },
        {
          label: 'Contributing',
          autogenerate: { directory: 'contributing' },
        },
      ],
    }),
    markdoc(),
  ],
});
