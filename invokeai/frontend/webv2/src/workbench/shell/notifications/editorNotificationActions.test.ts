import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';

import { getWidgetById, getWidgetsForRegion, registerWidgets } from '@workbench/widgetRegistry';
import { galleryWidgetManifest } from '@workbench/widgets/gallery/manifest';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { describe, expect, it } from 'vitest';

import {
  createEditorNotificationActionContext,
  type EditorNotificationActionSources,
} from './editorNotificationActions';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const i18n = createInstance();
await i18n.init({ initAsync: false, lng: 'en', resources: { en: { translation: englishCatalog } } });

type FocusMove = ['floating', string] | ['region', WidgetRegion, string | undefined];

const setup = (overrides: Partial<EditorNotificationActionSources> = {}) => {
  const store = createWorkbenchStore();
  const focusMoves: FocusMove[] = [];
  const context = createEditorNotificationActionContext({
    focus: {
      focusFloating: (instanceId) => focusMoves.push(['floating', instanceId]),
      focusRegion: (region, typeId) => focusMoves.push(['region', region, typeId]),
    },
    getWidgetById,
    getWidgetsForRegion,
    queries: store.queries,
    t: i18n.t,
    widgets: store.commands.widgets,
    ...overrides,
  });
  const project = (): Project => store.getSnapshot().activeProject;
  const frontOf = (region: WidgetRegion) => project().widgetInstances[project().widgetRegions[region].activeInstanceId];

  return { context, focusMoves, frontOf, project, store };
};

describe('editor notification actions', () => {
  it('names enabled widgets as the Gallery names itself and leaves out widgets this editor cannot show', () => {
    const [disabledGallery] = registerWidgets([{ ...galleryWidgetManifest, allowedRegions: [] }]);
    const { context } = setup();
    const { context: withDisabled } = setup({ getWidgetById: () => disabledGallery });

    expect(disabledGallery?.status).toBe('disabled');
    expect(context.getWidgetLabel('gallery')).toBe(i18n.t('widgets.labels.gallery'));
    expect(context.getWidgetLabel('not-installed')).toBeNull();
    expect(withDisabled.getWidgetLabel('gallery')).toBeNull();
  });

  it('focuses a widget already showing without changing the layout', () => {
    const { context, focusMoves, frontOf, project } = setup();
    const before = project().widgetRegions;

    expect(frontOf('right')?.typeId).toBe('gallery');
    context.openWidget('gallery');

    expect(project().widgetRegions).toBe(before);
    expect(focusMoves).toEqual([['region', 'right', 'gallery']]);
  });

  it('brings a docked widget behind another tab to the front of its own region', () => {
    const { context, focusMoves, frontOf } = setup();

    context.openWidget('queue');

    expect(frontOf('right')?.typeId).toBe('queue');
    expect(focusMoves).toEqual([['region', 'right', 'queue']]);
  });

  it('raises a floating window instead of docking another copy', () => {
    const { context, focusMoves, project, store } = setup();
    store.commands.widgets.float('queue');

    context.openWidget('queue');

    expect(Object.keys(project().floatingWidgets ?? {})).toEqual(['queue']);
    expect(focusMoves).toEqual([['floating', 'queue']]);
  });

  it('opens a widget the layout does not hold and focuses where it went', () => {
    const { context, focusMoves, frontOf, project } = setup();
    const holdsWorkflow = Object.values(project().widgetRegions).some((region) =>
      region.instanceIds.some((instanceId) => project().widgetInstances[instanceId]?.typeId === 'workflow')
    );

    expect(holdsWorkflow).toBe(false);
    context.openWidget('workflow');

    expect(frontOf('center')?.typeId).toBe('workflow');
    expect(focusMoves).toEqual([['region', 'center', 'workflow']]);
  });

  it('knows which projects are still open', () => {
    const { context, project } = setup();

    expect(context.isProjectOpen(project().id)).toBe(true);
    expect(context.isProjectOpen('closed-project')).toBe(false);
  });
});
