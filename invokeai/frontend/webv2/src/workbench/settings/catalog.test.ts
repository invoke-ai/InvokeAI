import type { SettingDefinition, SettingsContribution } from '@platform/ui/settings/contracts';
import type { WidgetManifest } from '@workbench/widgetContracts';

import { createInstance } from 'i18next';
import { EyeIcon } from 'lucide-react';
import { describe, expect, it, vi } from 'vitest';

import { buildSettingsCatalog, getAvailableSettings, searchSettings } from './catalog';

const field = (id: string, overrides: Partial<SettingDefinition> = {}): SettingDefinition =>
  ({ id, kind: 'boolean', label: id, scope: 'preference', ...overrides }) as SettingDefinition;
const contribution = (id: string, fields: readonly SettingDefinition[]): SettingsContribution => ({
  fields,
  id,
  label: id,
  load: vi.fn(() => Promise.resolve({ Field: () => null })),
});
const widget = (settings: SettingsContribution): WidgetManifest => ({
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: EyeIcon,
  id: 'preview',
  label: 'Preview widget',
  load: vi.fn(() => Promise.resolve({ view: () => null })),
  settings,
  version: 1,
});

const i18n = createInstance();
await i18n.init({
  lng: 'fr',
  resources: {
    fr: {
      translation: {
        section: 'Aperçu',
        label: 'Arrière-plan',
        description: 'Afficher les zones transparentes',
        group: 'Visibilité',
        option: 'Damier',
      },
    },
  },
});

describe('settings catalog composition', () => {
  it('builds and searches metadata without loading editors or widget implementations', () => {
    const application = contribution('appearance', [field('motion')]);
    const previewSettings = contribution('preview', [field('filmstrip')]);
    const preview = widget(previewSettings);
    const sections = buildSettingsCatalog([application], [preview]);

    expect(searchSettings(sections, 'filmstrip', i18n.t).map((section) => section.id)).toEqual(['preview']);
    expect(application.load).not.toHaveBeenCalled();
    expect(previewSettings.load).not.toHaveBeenCalled();
    expect(preview.load).not.toHaveBeenCalled();
  });

  it('merges application fields with their widget section while retaining field bindings and widget identity', async () => {
    const previewSettings = contribution('preview', [field('filmstrip', { scope: 'instance' })]);
    previewSettings.label = 'Preview preferences';
    const projectSettings = contribution('preview', [field('antialias', { scope: 'project' })]);
    const preview = widget(previewSettings);
    const sections = buildSettingsCatalog([projectSettings], [preview]);

    expect(sections).toHaveLength(1);
    const section = sections[0]!;
    expect(section).toMatchObject({
      group: 'widgets',
      icon: EyeIcon,
      id: 'preview',
      label: 'Preview preferences',
      widgetId: 'preview',
    });
    expect(section.entries.map((entry) => entry.field)).toEqual([previewSettings.fields[0], projectSettings.fields[0]]);
    const [instanceEntry, projectEntry] = section.entries;
    await instanceEntry!.resource.load();
    expect(previewSettings.load).toHaveBeenCalledOnce();
    expect(projectSettings.load).not.toHaveBeenCalled();
    await projectEntry!.resource.load();
    expect(projectSettings.load).toHaveBeenCalledOnce();
  });

  it('rejects duplicate entry IDs inside a contribution and across a merged section', () => {
    expect(() => buildSettingsCatalog([contribution('appearance', [field('motion'), field('motion')])], [])).toThrow(
      'Duplicate or empty setting appearance.motion'
    );
    expect(() =>
      buildSettingsCatalog(
        [contribution('preview', [field('filmstrip')])],
        [widget(contribution('preview', [field('filmstrip')]))]
      )
    ).toThrow('Duplicate or empty setting preview.filmstrip');
  });

  it('rejects empty entry IDs and quick placements that do not belong to the contribution', () => {
    expect(() => buildSettingsCatalog([contribution('appearance', [field('')])], [])).toThrow(
      'Duplicate or empty setting appearance.'
    );
    const previewSettings = contribution('preview', [field('filmstrip')]);
    previewSettings.quick = ['antialias'];
    expect(() =>
      buildSettingsCatalog([contribution('preview', [field('antialias')])], [widget(previewSettings)])
    ).toThrow('Unknown quick setting preview.antialias');
  });
});

describe('settings search', () => {
  const display = field('display', {
    description: (t) => t('description'),
    group: (t) => t('group'),
    keywords: 'checkerboard alpha',
    kind: 'select',
    label: (t) => t('label'),
    options: [{ label: (t) => t('option'), value: 'checker' }],
  });
  const previewSettings = contribution('preview', [
    display,
    field('vocabulary', { kind: 'custom', label: 'Vocabulary editor', keywords: 'semantic categories' }),
  ]);
  previewSettings.label = (t) => t('section');
  const sections = buildSettingsCatalog([], [widget(previewSettings)]);
  const resultIds = (query: string) =>
    searchSettings(sections, query, i18n.t).flatMap((section) => section.entries.map((entry) => entry.field.id));

  it.each(['ARRIERE', 'transparentes', 'visibilite', 'damier', 'checkerboard', 'alpha'])(
    'finds localized metadata and explicit aliases: %s',
    (query) => {
      expect(resultIds(query)).toEqual(['display']);
    }
  );

  it('requires every word across section, label, description, group, options, and aliases', () => {
    expect(resultIds(' APERCU   arriere transparentes VISIBILITE damier ALPHA ')).toEqual(['display']);
    expect(resultIds('arriere missing')).toEqual([]);
  });

  it('keeps custom editors searchable destinations and preserves catalog order', () => {
    expect(resultIds('semantic editor')).toEqual(['vocabulary']);
    expect(resultIds('apercu')).toEqual(['display', 'vocabulary']);
    expect(resultIds('   ')).toEqual(['display', 'vocabulary']);
    expect(searchSettings(sections, 'categories', i18n.t)[0]!.entries[0]!.field.kind).toBe('custom');
    expect(previewSettings.load).not.toHaveBeenCalled();
  });
});

describe('settings availability', () => {
  const sections = buildSettingsCatalog(
    [
      contribution('appearance', [field('motion')]),
      contribution('project', [field('cpuNoise', { scope: 'project' })]),
      contribution('preview', [field('antialias', { scope: 'project' })]),
      contribution('server', [field('devices', { scope: 'server' })]),
    ],
    [widget(contribution('preview', [field('filmstrip', { scope: 'instance' }), field('confirm')]))]
  );
  const shown = (availability: Parameters<typeof getAvailableSettings>[1]) =>
    getAvailableSettings(sections, availability).map((section) => [
      section.id,
      section.entries.map((entry) => entry.field.id),
    ]);

  it('leaves out project and widget settings where no project is open, and sections left empty', () => {
    expect(shown({ canManageAppConfig: true, widgetTypeIds: null })).toEqual([
      ['appearance', ['motion']],
      ['preview', ['confirm']],
      ['server', ['devices']],
    ]);
  });

  it('keeps widget settings only for widgets the open project contains', () => {
    expect(shown({ canManageAppConfig: true, widgetTypeIds: new Set(['gallery']) })).toEqual([
      ['appearance', ['motion']],
      ['project', ['cpuNoise']],
      ['preview', ['confirm', 'antialias']],
      ['server', ['devices']],
    ]);
    expect(shown({ canManageAppConfig: true, widgetTypeIds: new Set(['preview']) })[2]).toEqual([
      'preview',
      ['filmstrip', 'confirm', 'antialias'],
    ]);
  });

  it('shows the server section only to accounts that can manage the app configuration', () => {
    expect(shown({ canManageAppConfig: false, widgetTypeIds: null }).map(([id]) => id)).not.toContain('server');
  });
});
