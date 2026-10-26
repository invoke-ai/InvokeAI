// @ts-check
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';

// Plugins
import starlightLinksValidator from 'starlight-links-validator';
import starlightLlmsText from 'starlight-llms-txt';
import starlightChangelogs from 'starlight-changelogs';
import starlightSidebarTopics from 'starlight-sidebar-topics';
import { rehypePrefixBaseToRootLinks } from './plugins/rehype-prefix-base-to-root-links.mjs';
import { remarkLocalizeAssets } from './plugins/remark-localize-assets.mjs';
import { remarkSortPrefixSlug } from './plugins/remark-sort-prefix-slug.mjs';
import { rehypeTableScroll } from './plugins/rehype-table-scroll.mjs';
import { themeTokens, THEME_TOKENS_MODULE_ID } from './plugins/theme-tokens.mjs';
import starlightMarkdown from 'starlight-markdown';

// Configs
import { locales } from './src/config/locales.mjs';
import {
  createHeadConfig,
  createRedirects,
  sidebarTopics,
  sidebarTopicsOptions,
  socialConfig,
} from './src/config';
import * as appThemes from '../invokeai/frontend/webv2/src/platform/ui/theme/themes.ts';

const appThemesPath = new URL('../invokeai/frontend/webv2/src/platform/ui/theme/themes.ts', import.meta.url);

// Deployment target: 'custom' (default, custom domain at v7.invoke.ai) or 'ghpages'
// (GitHub Pages project URL at invoke-ai.github.io/InvokeAI). Drive site/base from this
// so the same source can be deployed to either target.
const deployTarget = process.env.DEPLOY_TARGET ?? 'custom';
const isGhPages = deployTarget === 'ghpages';
const enableAnalytics = process.env.ENABLE_ANALYTICS === 'true';
const base = isGhPages ? '/InvokeAI' : '';
const site = isGhPages ? 'https://invoke-ai.github.io' : 'https://v7.invoke.ai';

const docsDir = fileURLToPath(new URL('./src/content/docs', import.meta.url));
const redirects = createRedirects(base);
const head = createHeadConfig({ base, enableAnalytics, isGhPages, site });

/**
 * Runs the docs route middleware before starlight-sidebar-topics narrows the sidebar to the current topic; the
 * `routeMiddleware` option would run it after.
 * @type {import('@astrojs/starlight/types').StarlightPlugin}
 */
const docsRouteData = {
  name: 'invoke-docs-route-data',
  hooks: {
    'config:setup'({ addRouteMiddleware }) {
      addRouteMiddleware({ entrypoint: './src/route-data.ts', order: 'pre' });
    },
  },
};

/**
 * starlight-changelogs builds every release page once per locale, and has no option to limit that. Releases are English
 * GitHub content, so its hooks see only the root locale: release pages exist at the root only (see `ROOT_ONLY_PATHS`).
 * @type {() => import('@astrojs/starlight/types').StarlightPlugin}
 */
const starlightChangelogsAtRoot = () => {
  const plugin = starlightChangelogs();
  return {
    ...plugin,
    hooks: {
      ...plugin.hooks,
      'config:setup'(options) {
        const config = { ...options.config, locales: { root: locales.root } };
        return plugin.hooks['config:setup']?.({ ...options, config });
      },
    },
  };
};

// https://astro.build/config
export default defineConfig({
  site,
  base: base || undefined,
  markdown: {
    processor: unified({
      remarkPlugins: [
        [remarkSortPrefixSlug, { docsDir }],
        [remarkLocalizeAssets, { docsDir }],
      ],
      rehypePlugins: [[rehypePrefixBaseToRootLinks, { base }], rehypeTableScroll],
    }),
  },
  integrations: [
    starlight({
      // Content
      // Starlight shows the default language's title where a locale has none.
      title: {
        en: 'InvokeAI Documentation',
      },
      favicon: 'favicon.svg',
      editLink: {
        baseUrl: 'https://github.com/invoke-ai/InvokeAI/edit/main/docs',
      },
      head,
      lastUpdated: true,
      defaultLocale: 'root',
      locales,
      social: socialConfig,
      tableOfContents: {
        maxHeadingLevel: 4,
      },
      customCss: [
        THEME_TOKENS_MODULE_ID,
        '@fontsource-variable/inter',
        '@fontsource-variable/roboto-mono',
        './src/styles/custom.css',
      ],
      expressiveCode: {
        styleOverrides: {
          // The inner radius: the frame's 1px border brings its outer radius to the content-container radius.
          borderRadius: 'calc(var(--docs-radius-container) - 1px)',
          focusBorder: 'var(--docs-accent)',
        },
      },
      components: {
        ThemeProvider: './src/lib/components/ForceDarkTheme.astro',
        ThemeSelect: './src/lib/components/EmptyComponent.astro',
        Header: './src/lib/components/Header.astro',
        Search: './src/lib/components/Search.astro',
        // The mobile drawer's language picker (in Starlight's `MobileMenuFooter`); `Header` shows another copy.
        LanguageSelect: './src/lib/components/LanguageMenu.astro',
        Sidebar: './src/lib/components/Sidebar.astro',
        PageTitle: './src/lib/components/PageTitle.astro',
        MarkdownContent: './src/lib/components/MarkdownContent.astro',
        PageSidebar: './src/lib/components/PageSidebar.astro',
        Pagination: './src/lib/components/Pagination.astro',
        Footer: './src/lib/components/Footer.astro',
        PageFrame: './src/layouts/PageFrameExtended.astro',
      },
      plugins: [
        // Before starlight-sidebar-topics: see `docsRouteData`.
        docsRouteData,
        starlightSidebarTopics(sidebarTopics, sidebarTopicsOptions),
        starlightLinksValidator({
          errorOnRelativeLinks: false,
          errorOnLocalLinks: false,
          // A translated page may link into its own locale (`/de/…`), where pages not translated yet are fallback
          // routes. They are built, but the validator rejects them by default.
          errorOnFallbackPages: false,
        }),
        starlightLlmsText(),
        starlightChangelogsAtRoot(),
        // Each docs page's Markdown source at `<page>/index.md`, for the page actions in the right rail.
        starlightMarkdown(),
      ],
    }),
    themeTokens({ themesModule: appThemes, themesPath: appThemesPath }),
  ],
  redirects,
});
