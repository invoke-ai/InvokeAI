// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
import markdoc from '@astrojs/markdoc';
// import { strategy } from 'sharp';
// import rehypeExternalLinks from 'rehype-external-links';

// https://astro.build/config
export default defineConfig({
  site: 'https://invoke-ai.github.io',
  base: '/InvokeAI',
  integrations: [
    starlight({
      // Content
      title: {
        en: 'InvokeAI Documentation',
      },
      logo: {
        src: './src/assets/invoke-icon-wide.svg',
        alt: 'InvokeAI Logo',
        replacesTitle: true,
      },
      favicon: './src/assets/invoke-icon.svg',
      editLink: {
        baseUrl: 'https://github.com/invoke-ai/InvokeAI/edit/main/docs',
      },
      defaultLocale: 'root',
      locales: {
        root: {
          label: 'English',
          lang: 'en',
        },
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/invoke-ai/InvokeAI',
        },
        {
          icon: 'discord',
          label: 'Discord',
          href: 'https://discord.gg/ZmtBAhwWhy',
        },
      ],
      tableOfContents: {
        maxHeadingLevel: 4,
      },
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource-variable/roboto-mono',
        './src/styles/custom.css',
      ],
      sidebar: [
        {
          label: 'Start Here',
          autogenerate: { directory: 'start-here' },
        },
        {
          label: 'Configuration',
          autogenerate: { directory: 'configuration' },
        },
        {
          label: 'Concepts',
          autogenerate: { directory: 'concepts' },
        },
        {
          label: 'Features',
          autogenerate: { directory: 'features' },
        },
        {
          label: 'Workflows',
          autogenerate: { directory: 'workflows' },
        },
        {
          label: 'Development',
          autogenerate: { directory: 'development', collapsed: true },
        },
        {
          label: 'Contributing',
          autogenerate: { directory: 'contributing' },
          collapsed: true,
        },
        {
          label: 'Troubleshooting',
          autogenerate: { directory: 'troubleshooting' },
        },
      ],
      components: {
        ThemeProvider: './src/lib/components/ForceDarkTheme.astro',
        ThemeSelect: './src/lib/components/EmptyComponent.astro',
      },
    }),
    markdoc(),
  ],
});
