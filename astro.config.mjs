// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsText from 'starlight-llms-txt';
import starlightChangelogs, { makeChangelogsSidebarLinks } from 'starlight-changelogs';
// import starlightContextualMenu from 'starlight-contextual-menu';

// Deployment target: 'custom' (default, custom domain at invoke.ai) or 'ghpages'
// (GitHub Pages project URL at invoke-ai.github.io/InvokeAI). Drive site/base from this
// so the same source can be deployed to either target.
const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const isGhPages = deployTarget === 'ghpages';

// https://astro.build/config
export default defineConfig({
  site: isGhPages ? 'https://invoke-ai.github.io' : 'https://invoke.ai',
  base: isGhPages ? '/InvokeAI' : undefined,
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
      favicon: '/favicon.svg',
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
        // The links validator is skipped for the ghpages target because content uses
        // root-absolute links (e.g. /concepts/...) that don't include the /InvokeAI base.
        // Production (custom domain) still enforces link validation.
        ...(isGhPages
          ? []
          : [
              starlightLinksValidator({
                errorOnRelativeLinks: false,
                errorOnLocalLinks: false,
              }),
            ]),
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
