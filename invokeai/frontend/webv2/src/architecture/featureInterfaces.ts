/**
 * Fail-closed public-surface registry. A feature absent from this map has NO
 * public modules; a top-level module absent from its list is private.
 * `index` is implicitly public for every registered feature.
 * To publish a new entry module, add it here and cover it in dependencyPolicy.test.ts.
 */
export const FEATURE_PUBLIC_INTERFACES: Readonly<Record<string, readonly string[]>> = {
  gallery: ['contracts', 'queries', 'react', 'utility', 'widget'],
  generation: ['components', 'contracts', 'graph', 'react', 'settings', 'widget'],
  identity: [],
  models: ['react'],
  nodes: [],
  queue: ['contracts', 'menu', 'react', 'utility', 'widget'],
  upscale: ['widget'],
  workflow: ['contracts', 'graph', 'preview', 'react', 'utility', 'widget'],
};
