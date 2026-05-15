// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Plugins
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsText from 'starlight-llms-txt';
import starlightChangelogs from 'starlight-changelogs';
import { rehypePrefixBaseToRootLinks } from './plugins/rehype-prefix-base-to-root-links.mjs';
import starlightContextualMenu from 'starlight-contextual-menu';

// Configs
import { createRedirects, sidebarConfig, socialConfig } from './src/config';

// Deployment target: 'custom' (default, custom domain at invoke.ai) or 'ghpages'
// (GitHub Pages project URL at invoke-ai.github.io/InvokeAI). Drive site/base from this
// so the same source can be deployed to either target.
const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const isGhPages = deployTarget === 'ghpages';
const enableAnalytics = process.env.ENABLE_ANALYTICS === 'true';
const base = isGhPages ? '/InvokeAI' : '';

const redirects = createRedirects(base);

// https://astro.build/config
export default defineConfig({
  site: isGhPages ? 'https://invoke-ai.github.io' : 'https://invoke.ai',
  base: base || undefined,
  markdown: {
    rehypePlugins: [[rehypePrefixBaseToRootLinks, { base }]],
  },
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
      favicon: 'favicon.svg',
      editLink: {
        baseUrl: 'https://github.com/invoke-ai/InvokeAI/edit/main/docs',
      },
      head:
        enableAnalytics && !isGhPages
          ? [
              {
                tag: 'script',
                attrs: {
                  async: true,
                  src: 'https://plausible.tracking.events/js/pa-BHcumuOemKz4XIQeWkTn4.js',
                },
              },
              {
                tag: 'script',
                content:
                  'window.plausible=window.plausible||function(){(plausible.q=plausible.q||[]).push(arguments)},plausible.init=plausible.init||function(i){plausible.o=i||{}};plausible.init()',
              },
            ]
          : [],
      defaultLocale: 'root',
      locales: {
        root: {
          label: 'English',
          lang: 'en',
        },
      },
      social: socialConfig,
      tableOfContents: {
        maxHeadingLevel: 4,
      },
      customCss: [
        '@fontsource-variable/inter',
        '@fontsource-variable/roboto-mono',
        './src/styles/custom.css',
      ],
      sidebar: sidebarConfig,
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
        starlightContextualMenu({
          actions: ['copy', 'view', 'chatgpt', 'claude'],
        }),
      ],
    }),
  ],
  redirects,
});
