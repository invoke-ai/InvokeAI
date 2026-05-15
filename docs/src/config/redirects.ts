import type { AstroConfig } from 'astro';

type RedirectsConfig = AstroConfig['redirects'];

const redirects: RedirectsConfig = {
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
  '/features/lasso-tool': '/features/canvas/lasso-tool',
  '/features/shapes-tool': '/features/canvas/shapes-tool',
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
  '/contributing/contribution_guides/newContributorChecklist':
    '/contributing/new-contributor-guide',
  '/contributing/dev-environment': '/development/setup/dev-environment',
  '/contributing/frontend': '/development/front-end',
  '/contributing/frontend/state-management':
    '/development/front-end/state-management',
  '/contributing/frontend/workflows': '/development/front-end/workflows',
};

function createRedirects(base: string): RedirectsConfig {
  return Object.fromEntries(
    Object.entries(redirects).map(([from, to]) => [from, base + to]),
  );
}

export { createRedirects };
