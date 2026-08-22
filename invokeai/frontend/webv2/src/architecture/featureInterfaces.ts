/**
 * Fail-closed public-surface registry. A feature absent from this map has NO
 * public modules; a top-level module absent from its list is private.
 * `index` is implicitly public for every registered feature.
 * To publish a new entry module, add it here and cover it in dependencyPolicy.test.ts.
 */
export const FEATURE_PUBLIC_INTERFACES: Readonly<Record<string, readonly string[]>> = {
  gallery: ['contracts', 'launchpad', 'paletteSearch', 'queries', 'react', 'utility', 'widget'],
  generation: [
    'components',
    'contracts',
    'graph',
    'preview',
    'prompts',
    'queries',
    'react',
    'runtime',
    'settings',
    'widget',
  ],
  identity: [],
  models: ['launchpad', 'react'],
  nodes: [],
  queue: ['contracts', 'devices', 'launchpad', 'menu', 'queries', 'react', 'reveal', 'utility', 'widget'],
  upscale: ['widget'],
  video: [],
  workflow: ['contracts', 'graph', 'paletteSearch', 'preview', 'queries', 'react', 'utility', 'widget'],
};
