/**
 * Unregistered features and unlisted modules are private; index is public for registered features. Cover new
 * entries in dependencyPolicy.test.ts.
 */
export const FEATURE_PUBLIC_INTERFACES: Readonly<Record<string, readonly string[]>> = {
  gallery: [
    'settingsContribution',
    'contracts',
    'launchpad',
    'mediaSlot',
    'paletteSearch',
    'picker',
    'queries',
    'react',
    'utility',
    'widget',
  ],
  fonts: ['contracts', 'launchpad', 'react', 'runtime'],
  generation: [
    'canvasGraph',
    'canvasProcessingSize',
    'components',
    'contracts',
    'graph',
    'preview',
    'prompts',
    'queries',
    'react',
    'runtime',
    'settings',
    'systemPrompts',
    'widget',
  ],
  identity: [],
  intermediates: ['holdLease', 'settingsContribution'],
  models: ['launchpad', 'react'],
  nodes: [],
  queue: ['contracts', 'devices', 'launchpad', 'menu', 'queries', 'react', 'reveal', 'utility', 'widget'],
  upscale: ['widget'],
  video: ['widget'],
  workflow: ['contracts', 'generators', 'graph', 'paletteSearch', 'preview', 'queries', 'react', 'utility', 'widget'],
};
