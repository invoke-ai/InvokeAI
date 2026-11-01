import type { SettingDefinition, SettingsContribution, SettingsText } from '@platform/ui/settings/contracts';

import { intermediatesSettingsContribution } from '@features/intermediates/settingsContribution';
import { WORKBENCH_LANGUAGE_OPTIONS } from '@platform/i18n/languages';

const text =
  (key: string, fallback?: string): SettingsText =>
  (t) =>
    t(`settings.catalog.${key}`, { defaultValue: fallback });
const preference = (id: string, label: string, keywords?: string): SettingDefinition => ({
  id,
  label: text(`${id}.label`, label),
  description: text(`${id}.description`),
  keywords,
  kind: 'boolean',
  scope: 'preference',
});
const custom = (
  id: string,
  label: string,
  scope: SettingDefinition['scope'] = 'preference',
  options: { fill?: boolean; framed?: boolean } = {}
): SettingDefinition => ({
  id,
  label: text(`${id}.label`, label),
  description: text(`${id}.description`),
  kind: 'custom',
  scope,
  ...options,
});
const section = (id: string, label: string, fields: readonly SettingDefinition[]): SettingsContribution => ({
  id,
  label: text(`sections.${id}`, label),
  fields,
  load: () => import('./ApplicationSettingField').then((module) => ({ Field: module.ApplicationSettingField })),
});

export const appearanceSettings = section('appearance', 'Appearance', [
  custom('themeId', 'Theme'),
  {
    id: 'language',
    label: text('language.label', 'Language'),
    kind: 'select',
    scope: 'preference',
    options: WORKBENCH_LANGUAGE_OPTIONS,
  },
  preference('reduceMotion', 'Reduce motion', 'animation'),
  preference('highContrast', 'High contrast', 'accessibility a11y readability'),
  preference('showFocusRegionHighlight', 'Highlight focused regions', 'panel outline'),
  {
    id: 'promptFontSize',
    label: text('promptFontSize.label', 'Prompt text size'),
    description: text('promptFontSize.description'),
    keywords: 'font generate prompt editor',
    kind: 'select',
    scope: 'preference',
    options: [
      { value: 'small', label: text('options.small', 'Small') },
      { value: 'default', label: text('options.default', 'Default') },
      { value: 'large', label: text('options.large', 'Large') },
      { value: 'larger', label: text('options.larger', 'Larger') },
    ],
  },
]);
export const behaviorSettings = section('behavior', 'Behavior', [
  {
    id: 'startupDestination',
    label: text('startupDestination.label', 'On startup, open'),
    description: text('startupDestination.description'),
    keywords: 'launch start open resume recent project launchpad home',
    kind: 'select',
    scope: 'preference',
    options: [
      { value: 'launchpad', label: text('options.launchpad', 'Launchpad') },
      { value: 'recentProject', label: text('options.recentProject', 'Most recent project') },
    ],
  },
  preference('autoSwitchInvocationRoute', 'Invoke follows what you edit', 'route source destination'),
  preference('enableInformationalPopovers', 'Enable informational popovers', 'help tooltips'),
  preference('enableModelDescriptions', 'Show descriptions in model pickers', 'models dropdowns'),
  preference('notifyOnEnqueue', 'Notify when queued', 'notifications queue'),
  preference('preferNumericAttentionStyle', 'Prefer numeric attention style', 'generate prompt editor'),
  preference('showPromptSyntaxHighlighting', 'Highlight prompt syntax', 'generate prompt editor'),
]);
export const hotkeysSettings = section('hotkeys', 'Keyboard shortcuts', [
  custom('hotkeys', 'Keyboard shortcuts', 'preference', { fill: true }),
]);
export const galleryPreferenceSettings = section('gallery', 'Gallery', [
  preference('confirmImageDeletion', 'Confirm image deletion', 'delete safety'),
]);
export const previewPreferenceSettings = section('preview', 'Preview', [
  preference('antialiasProgressImages', 'Antialias progress images'),
]);
export const workflowSettings = section('workflow', 'Workflow', [
  {
    id: 'workflowEdgeStyle',
    label: text('workflowEdgeStyle.label', 'Connection style'),
    description: text('workflowEdgeStyle.description'),
    kind: 'select',
    scope: 'preference',
    options: [
      { value: 'curved', label: text('options.curved', 'Curved') },
      { value: 'square', label: text('options.square', 'Square') },
    ],
  },
  preference(
    'workflowEdgesBehindNodes',
    'Keep highlighted connections behind nodes',
    'workflow edges wires links z-order'
  ),
  preference('workflowSnapToGrid', 'Always snap to grid', 'workflow nodes'),
  {
    id: 'workflowFormSectionGapPx',
    label: text('workflowFormSectionGapPx.label', 'Spacing between form sections (px)'),
    description: text(
      'workflowFormSectionGapPx.description',
      'Adjust the gap between collapsible workflow form sections.'
    ),
    kind: 'slider',
    scope: 'preference',
    min: 4,
    max: 36,
    step: 1,
  },
  preference('workflowGroupNodesByCategory', 'Group node search by category', 'workflow nodes add search categories'),
  preference('workflowShowMinimap', 'Show minimap'),
  preference('workflowValidateConnections', 'Validate connections', 'workflow edges'),
]);
export const queueSettings = section('queue', 'Queue', [
  {
    id: 'queueJobsScope',
    label: text('queueJobsScope.label', 'Show jobs from'),
    description: text('queueJobsScope.description'),
    kind: 'select',
    scope: 'preference',
    options: [
      { value: 'active-project', label: text('options.activeProject', 'Active project') },
      { value: 'all', label: text('options.all', 'All projects') },
    ],
  },
]);
export const imageMapVocabularySettings = section('imageMap', 'Image Map', [
  custom('imageMapVocabulary', 'Vocabulary', 'server'),
]);
export const developerSettings = section('developer', 'Developer', [
  preference('developerLogEnabled', 'Record diagnostic logs', 'debug console'),
  {
    id: 'developerLogLevel',
    label: text('developerLogLevel.label', 'Log level'),
    kind: 'select',
    scope: 'preference',
    options: ['trace', 'debug', 'info', 'warn', 'error', 'fatal'].map((value) => ({
      value,
      label: text(`options.${value}`, value.charAt(0).toUpperCase() + value.slice(1)),
    })),
  },
  preference('developerConsoleOutputEnabled', 'Mirror logs to the browser console', 'console devtools'),
  preference('developerPerformanceTimingsEnabled', 'Collect performance timings', 'diagnostics'),
  custom('developerLogNamespaces', 'Log namespaces'),
  custom('developerLoggingReset', 'Reset logging defaults', 'preference'),
]);
export const serverSettings = section('server', 'Server', [
  custom('generationDevices', 'Generation devices', 'server'),
  { ...custom('imageStorage', 'Image storage', 'server'), keywords: 'images folder subfolder layout move recover' },
  custom('maxQueueHistory', 'Queue history', 'server'),
]);
export const workspaceSettings = section('workspace', 'Data & workspace', [
  custom('workspaceActions', 'Data & workspace', 'none'),
  {
    ...custom('clearSavedData', 'Clear saved data', 'none', { framed: true }),
    keywords: 'delete reset erase wipe projects',
  },
]);
export const aboutSettings = section('about', 'About', [custom('about', 'About Invoke', 'none')]);

export const applicationSettingsContributions = [
  appearanceSettings,
  behaviorSettings,
  hotkeysSettings,
  galleryPreferenceSettings,
  previewPreferenceSettings,
  imageMapVocabularySettings,
  developerSettings,
  serverSettings,
  workspaceSettings,
  intermediatesSettingsContribution,
  aboutSettings,
];
/** These are the same descriptors rendered in settings and used by palette preference actions. */
export const preferenceSettingsFields = [...applicationSettingsContributions, workflowSettings, queueSettings].flatMap(
  (contribution) =>
    contribution.fields
      .filter((field) => field.scope === 'preference' && field.kind !== 'custom')
      .map((field) => ({ field, sectionId: contribution.id }))
);
