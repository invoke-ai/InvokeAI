import type { WidgetRegion } from '@workbench/layoutContracts';
import type { Project } from '@workbench/projectContracts';

import { getWidgetById, getWidgetsForRegion, registerWidgets } from '@workbench/widgetRegistry';
import { galleryWidgetManifest } from '@workbench/widgets/gallery/manifest';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  createEditorNotificationActionContext,
  type EditorNotificationActionSources,
} from './editorNotificationActions';

const englishCatalog = Object.values(
  import.meta.glob('../../../../public/locales/en.json', { eager: true, import: 'default' })
)[0] as Record<string, unknown>;
const reveal = vi.hoisted(() => ({ requested: [] as number[] }));
vi.mock('@features/queue/reveal', () => ({
  requestQueueItemReveal: (itemId: number) => reveal.requested.push(itemId),
}));

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
    notifications: store.commands.notifications,
    queries: store.queries,
    retryQueueItems: () => Promise.resolve([]),
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

  it('shows a queue item by opening the queue at it', () => {
    reveal.requested = [];
    const { context, frontOf } = setup();

    context.revealQueueItem(41);

    expect(frontOf('right')?.typeId).toBe('queue');
    expect(reveal.requested).toEqual([41]);
  });

  it('reports a retry it queued with a way to the rerun, and says so when the server queued none or failed', async () => {
    const queued = setup({ retryQueueItems: () => Promise.resolve([99]) });
    const none = setup({ retryQueueItems: () => Promise.resolve([]) });
    const failed = setup({ retryQueueItems: () => Promise.reject(new Error('Queue item 43 not found')) });

    queued.context.retryQueueItem({ itemIds: [41] });
    none.context.retryQueueItem({ itemIds: [42] });
    failed.context.retryQueueItem({ itemIds: [43] });

    await vi.waitFor(() => expect(failed.store.getSnapshot().notifications).toHaveLength(1));
    // Recorded as keys and values, so the notice follows the language.
    expect(queued.store.getSnapshot().notifications[0]).toMatchObject({
      actions: [{ itemId: 99, kind: 'reveal-queue-item' }],
      kind: 'success',
      messageKey: 'widgets.queue.retryQueuedDescription',
      messageValues: { count: 1, id: 41 },
      titleKey: 'widgets.queue.retryQueued',
    });
    expect(none.store.getSnapshot().notifications[0]).toMatchObject({
      kind: 'info',
      message: 'Queue item #42 is not a failed or canceled job the server can retry.',
      messageValues: { count: 1, id: 42 },
      titleKey: 'widgets.queue.retryNotQueued',
    });
    expect(failed.store.getSnapshot().notifications[0]).toMatchObject({
      kind: 'error',
      message: 'Queue item 43 not found',
      titleKey: 'widgets.queue.retryFailed',
    });
    // A failed attempt may be tried again; a done one is not offered again.
    expect(failed.context.canRetryQueueItem([43])).toBe(true);
    expect(queued.context.canRetryQueueItem([41])).toBe(false);
  });

  it('retries every failed item of a run in one request and says how many were queued again', async () => {
    const retryQueueItems = vi.fn(() => Promise.resolve([101, 102, 103, 104]));
    const { context, store } = setup({ retryQueueItems });
    store.commands.notifications.add({
      actions: [{ itemIds: [41, 42, 43, 44], kind: 'retry-queue-item' }],
      kind: 'error',
      title: 'Run failed',
    });

    context.retryQueueItem({ itemIds: [41, 42, 43, 44] });

    await vi.waitFor(() => expect(store.getSnapshot().notifications).toHaveLength(2));
    expect(retryQueueItems).toHaveBeenCalledOnce();
    expect(retryQueueItems).toHaveBeenCalledWith([41, 42, 43, 44]);
    const [report, failure] = store.getSnapshot().notifications;
    expect(report).toMatchObject({
      actions: [{ itemId: 101, kind: 'reveal-queue-item' }],
      message: '4 queue items were queued again from their saved sessions.',
    });
    // Several items asked for are counted, not named by the first.
    expect(report?.messageValues).toEqual({ count: 4 });
    // The notice that offered the retry no longer does.
    expect(failure).not.toHaveProperty('actions');
  });

  it('says how many of several items the server queued again when it skipped some', async () => {
    const { context, store } = setup({ retryQueueItems: () => Promise.resolve([101]) });

    context.retryQueueItem({ itemIds: [41, 42, 43] });

    await vi.waitFor(() => expect(store.getSnapshot().notifications).toHaveLength(1));
    const [report] = store.getSnapshot().notifications;
    expect(report).toMatchObject({
      message:
        '1 of 3 queue items was queued again from its saved session; the others are not failed or canceled jobs the server can retry.',
      messageKey: 'widgets.queue.retryQueuedPartialDescription',
    });
    expect(report?.messageValues).toEqual({ count: 1, total: 3 });
  });

  it('queues a rerun once however often Retry is pressed, from any notice', async () => {
    const retryQueueItems = vi.fn(() => Promise.resolve([100]));
    const { context } = setup({ retryQueueItems });
    const { context: otherNotice } = setup({ retryQueueItems });

    context.retryQueueItem({ itemIds: [51, 52] });
    context.retryQueueItem({ itemIds: [51, 52] });
    otherNotice.retryQueueItem({ itemIds: [52, 51] });
    await vi.waitFor(() => expect(context.canRetryQueueItem([51, 52])).toBe(false));
    context.retryQueueItem({ itemIds: [51, 52] });

    expect(retryQueueItems).toHaveBeenCalledOnce();
  });

  it('says a Canvas rerun goes to another tab when its project is closed here', async () => {
    const { context, store } = setup({ retryQueueItems: () => Promise.resolve([101]) });

    context.retryQueueItem({ canvasProjectId: 'closed-project', itemIds: [61] });

    await vi.waitFor(() =>
      expect(store.getSnapshot().notifications[0]?.messageKey).toBe('widgets.queue.retryQueuedProjectClosedDescription')
    );
  });

  it('knows which projects are still open', () => {
    const { context, project } = setup();

    expect(context.isProjectOpen(project().id)).toBe(true);
    expect(context.isProjectOpen('closed-project')).toBe(false);
  });
});
