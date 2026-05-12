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
      favicon: 'favicon.svg',
      editLink: {
        baseUrl: 'https://github.com/invoke-ai/InvokeAI/edit/main/docs',
      },
      head: isGhPages
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
      ]
    }),
  ],
  redirects: {
    '/CODE_OF_CONDUCT': '/contributing/code-of-conduct',
    '/RELEASE': '/development/process/release-process',
    '/installation': '/start-here/installation',
    '/installation/docker': '/configuration/docker',
    '/installation/manual': '/start-here/manual',
    '/installation/models': '/concepts/models',
    '/installation/patchmatch': '/configuration/patchmatch',
    '/installation/quick_start': '/start-here/installation',
    '/installation/requirements': '/start-here/system-requirements',
    '/configuration': '/configuration/invokeai-yaml',
    '/features/low-vram/': '/configuration/low-vram-mode/',
    '/faq': '/troubleshooting/faq',
    '/help/SAMPLER_CONVERGENCE': '/concepts/parameters',
    '/help/diffusion': '/concepts/diffusion',
    '/help/gettingStartedWithAI': '/concepts/image-generation',
    '/nodes/NODES': '/workflows/editor-interface',
    '/nodes/NODES_MIGRATION_V3_V4': '/development/guides/api-development',
    '/nodes/comfyToInvoke': '/workflows/comfyui-migration',
    '/nodes/communityNodes': '/workflows/community-nodes',
    '/nodes/contributingNodes': '/development/guides/creating-nodes',
    '/nodes/invocation-api': '/development/guides/api-development',
    '/contributing/ARCHITECTURE': '/development/architecture/overview',
    '/contributing/DOWNLOAD_QUEUE': '/development/architecture/model-manager',
    '/contributing/HOTKEYS': '/features/hotkeys',
    '/contributing/INVOCATIONS': '/development/architecture/invocations',
    '/contributing/LOCAL_DEVELOPMENT': '/development/setup/dev-environment',
    '/contributing/MODEL_MANAGER': '/development/architecture/model-manager',
    '/contributing/NEW_MODEL_INTEGRATION': '/development/guides/models',
    '/contributing/PR-MERGE-POLICY': '/development/process/pr-merge-policy',
    '/contributing/TESTS': '/development/guides/tests',
    '/contributing/contribution_guides/development': '/development',
    '/contributing/contribution_guides/newContributorChecklist': '/contributing/new-contributor-guide',
    '/contributing/dev-environment': '/development/setup/dev-environment',
    '/contributing/frontend': '/development/front-end',
    '/contributing/frontend/state-management': '/development/front-end/state-management',
    '/contributing/frontend/workflows': '/development/front-end/workflows',
  }
});
