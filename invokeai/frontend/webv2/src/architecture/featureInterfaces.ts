/**
 * Unregistered features and unlisted modules are private; index is public for registered features. Cover new entries in
 * dependencyPolicy.test.ts.
 */
export const FEATURE_PUBLIC_INTERFACES: Readonly<Record<string, readonly string[]>> = {
  gallery: [
    'settingsContribution',
    'contracts',
    'launchpad',
    'mediaSlot',
    'mediaUsage',
    'paletteSearch',
    'picker',
    'queries',
    'react',
    'storage',
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
    'graphExport',
    'preview',
    'prompts',
    'queries',
    'react',
    'runtime',
    'settings',
    'systemPrompts',
    'widget',
  ],
  identity: ['launchpad'],
  intermediates: ['holdLease', 'settingsContribution'],
  models: ['architectures', 'architecturesStore', 'launchpad', 'react'],
  nodes: [],
  queue: ['contracts', 'devices', 'launchpad', 'menu', 'queries', 'react', 'reveal', 'utility', 'widget'],
  upscale: ['sourceMetadata', 'widget'],
  video: ['widget'],
  workflow: ['contracts', 'generators', 'graph', 'paletteSearch', 'preview', 'queries', 'react', 'utility', 'widget'],
};

/**
 * Public entries that pure cores may import. Each may reach only its own feature's `core/`, so a later re-export cannot
 * pull data, transport, UI or React into the cores that read it.
 */
export const PURE_FEATURE_ENTRIES: readonly string[] = ['features/models/architectures'];
