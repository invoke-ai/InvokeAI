// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsText from 'starlight-llms-txt';
import starlightChangelogs, { makeChangelogsSidebarLinks } from 'starlight-changelogs';
// import starlightContextualMenu from 'starlight-contextual-menu';

// https://astro.build/config
export default defineConfig({
  site: 'https://invoke.ai',
  // base is only needed if no custom domain is available, or if the site is hosted in a subdirectory
  // base: '/InvokeAI',
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
        {
          icon: 'youtube',
          label: 'YouTube',
          href: 'https://www.youtube.com/@invokeai',
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
          collapsed: true,
        },
        {
          label: 'Development',
          autogenerate: { directory: 'development', collapsed: true },
          collapsed: true,
        },
        {
          label: 'Contributing',
          autogenerate: { directory: 'contributing' },
          collapsed: true,
        },
        {
          label: 'Troubleshooting',
          autogenerate: { directory: 'troubleshooting' },
          collapsed: true,
        },
        {
          label: 'Releases',
          collapsed: true,
          items: [
            ...makeChangelogsSidebarLinks([
              {
                type: 'recent',
                base: 'releases',
              }
            ])
          ]
        }
      ],
      components: {
        ThemeProvider: './src/lib/components/ForceDarkTheme.astro',
        ThemeSelect: './src/lib/components/EmptyComponent.astro',
        Footer: './src/lib/components/Footer.astro',
        PageFrame: './src/layouts/PageFrameExtended.astro',
      },
      plugins: [
        starlightLinksValidator({
          errorOnRelativeLinks: false,
          errorOnLocalLinks: false,
        }),
        starlightLlmsText(),
        starlightChangelogs(),
        // starlightContextualMenu({
        //   actions: [
        //     'copy', 'view', 'chatgpt', 'claude'
        //   ]
        // }),
      ],
    }),
  ],
});
