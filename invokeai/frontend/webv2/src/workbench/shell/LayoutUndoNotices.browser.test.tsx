import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createEditorNotificationActionContext } from './notifications/editorNotificationActions';
import { startNotificationToasts } from './notifications/notificationToasts';

const workbench = vi.hoisted(() => ({ store: null as ReturnType<typeof createWorkbenchStore> | null }));

// A real store without the provider's persistence: the notices read and undo through it.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useOptionalWorkbenchCommands: () => workbench.store?.commands ?? null,
  useWorkbenchCommands: () => workbench.store!.commands,
  useWorkbenchQueries: () => workbench.store!.queries,
  useWorkbenchSubscription: () => workbench.store!.subscribe,
}));

const { LayoutUndoNotices } = await import('./LayoutUndoNotices');

await i18next.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let stop: (() => void) | null = null;

const store = () => workbench.store!;
const activeProject = () => store().getSnapshot().activeProject;

const toastNamed = (title: string): HTMLElement | null =>
  [...document.querySelectorAll<HTMLElement>('[data-part="root"][data-scope="toast"]')].find(
    (toast) => toast.querySelector('[data-part="title"]')?.textContent === title
  ) ?? null;

const openToast = async (title: string): Promise<HTMLElement> => {
  await expect.poll(() => toastNamed(title)?.getAttribute('data-state')).toBe('open');
  return toastNamed(title)!;
};

beforeEach(async () => {
  workbench.store = createWorkbenchStore();
  store().commands.layout.applyPreset('compose');
  stop = startNotificationToasts({
    actionContext: createEditorNotificationActionContext({
      focus: { focusFloating: () => {}, focusRegion: () => {} },
      getWidgetById,
      getWidgetsForRegion,
      notifications: store().commands.notifications,
      queries: store().queries,
      retryQueueItems: () => Promise.resolve([]),
      t: i18next.t,
      widgets: store().commands.widgets,
    }),
    getNotifications: () => store().getSnapshot().notifications,
    getPreferences: () => ({ notifyOnEnqueue: true }),
    subscribe: store().subscribe,
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <AppToaster />
        <LayoutUndoNotices />
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  stop?.();
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  workbench.store = null;
});

describe('LayoutUndoNotices', () => {
  it('reports a removed widget with an Undo that puts it back, and keeps the notice in the center', async () => {
    const region = (['left', 'right'] as const).find((candidate) =>
      activeProject().widgetRegions[candidate].instanceIds.includes('gallery')
    )!;

    await act(() => store().commands.widgets.toggle({ region, widgetId: 'gallery' }));
    const toast = await openToast('Gallery removed');

    expect(
      store()
        .getSnapshot()
        .notifications.map(({ title }) => title)
    ).toContain('Gallery removed');
    await act(() => toast.querySelector<HTMLButtonElement>('button:not([data-part="close-trigger"])')!.click());

    expect(activeProject().widgetRegions[region].instanceIds).toContain('gallery');
  });

  it('reports a preset switch as a toast only, saying where Invoke now runs when the switch moved it', async () => {
    expect(activeProject().invocation.sourceId).not.toBe('workflow');

    await act(() => store().commands.layout.applyPreset('automate'));
    const toast = await openToast('Switched to Workflow');

    expect(activeProject().invocation.sourceId).toBe('workflow');
    expect(toast.textContent).toContain('Invoke now runs Workflow');
    expect(
      store()
        .getSnapshot()
        .notifications.map(({ title }) => title)
    ).not.toContain('Switched to Workflow');

    await act(() => toast.querySelector<HTMLButtonElement>('button:not([data-part="close-trigger"])')!.click());
    expect(activeProject().layout.presetId).toBe('compose');
  });

  // Undone another way, the step is gone: an Undo still offered would do nothing.
  it('withdraws the toast once its step is undone by the layout undo command', async () => {
    const region = (['left', 'right'] as const).find((candidate) =>
      activeProject().widgetRegions[candidate].instanceIds.includes('gallery')
    )!;

    await act(() => store().commands.widgets.toggle({ region, widgetId: 'gallery' }));
    await openToast('Gallery removed');
    await act(() => store().commands.layout.undo());

    await expect.poll(() => toastNamed('Gallery removed')?.getAttribute('data-state') ?? 'gone').not.toBe('open');
    expect(activeProject().widgetRegions[region].instanceIds).toContain('gallery');
  });

  it('withdraws the toast once another step lands on top, so its Undo cannot take that work with it', async () => {
    const region = (['left', 'right'] as const).find((candidate) =>
      activeProject().widgetRegions[candidate].instanceIds.includes('gallery')
    )!;

    await act(() => store().commands.widgets.toggle({ region, widgetId: 'gallery' }));
    await openToast('Gallery removed');
    await act(() => store().commands.generation.toggleRoutingLock());

    await expect.poll(() => toastNamed('Gallery removed')?.getAttribute('data-state') ?? 'gone').not.toBe('open');
  });

  it('marks a step in a project not on screen as seen, without a toast now or later', async () => {
    const first = activeProject().id;
    const second = store().commands.projects.create().id;

    await act(() => store().commands.projects.switchTo(second));
    await act(() => store().commands.widgets.toggle({ projectId: first, region: 'right', widgetId: 'queue' }));
    await act(() => store().commands.projects.switchTo(first));
    await act(() => store().commands.generation.toggleRoutingLock());

    expect(toastNamed('Queue removed')).toBeNull();
  });

  it('stays quiet for layout steps that carry no notice, and for undoing', async () => {
    await act(() => store().commands.generation.toggleRoutingLock());
    await act(() => store().commands.layout.undo());

    expect(document.querySelectorAll('[data-scope="toast"][data-part="root"]')).toHaveLength(0);
  });
});
