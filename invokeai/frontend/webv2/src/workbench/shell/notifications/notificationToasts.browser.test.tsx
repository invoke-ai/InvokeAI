import type { Notify } from '@workbench/useNotify';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { useMountEffect } from '@platform/react/useMountEffect';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { AppToaster, toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { getWidgetById, getWidgetsForRegion } from '@workbench/widgetRegistry';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createEditorNotificationActionContext } from './editorNotificationActions';
import { startNotificationToasts } from './notificationToasts';
import { hasNotificationUndo } from './undoRegistry';

const workbench = vi.hoisted(() => ({ store: null as ReturnType<typeof createWorkbenchStore> | null }));

// `useNotify` records into whichever workbench it finds; here, a real store without the provider's persistence.
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useOptionalWorkbenchCommands: () => workbench.store?.commands ?? null,
}));

const { useNotify } = await import('@workbench/useNotify');

// The app's own instance and catalog: the no-editor fallback reads `i18next.t` directly.
await i18next.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
let stop: (() => void) | null = null;
const probe: { notify: Notify | null } = { notify: null };
const focusMoves: string[] = [];

/** `useNotify` is stable for a given workbench, so publishing it once at mount is enough. */
const NotifyProbe = () => {
  const notify = useNotify();

  useMountEffect(() => {
    probe.notify = notify;
  });

  return null;
};

const start = (store: ReturnType<typeof createWorkbenchStore>) =>
  startNotificationToasts({
    actionContext: createEditorNotificationActionContext({
      focus: {
        focusFloating: (instanceId) => focusMoves.push(`floating:${instanceId}`),
        focusRegion: (region, typeId) => focusMoves.push(`${region}:${typeId ?? ''}`),
      },
      getWidgetById,
      getWidgetsForRegion,
      queries: store.queries,
      t: i18next.t,
      widgets: store.commands.widgets,
    }),
    getNotifications: () => store.getSnapshot().notifications,
    getPreferences: () => ({ notifyOnEnqueue: true }),
    subscribe: store.subscribe,
  });

/** Mounts the toaster and a `useNotify` caller, inside an editor unless `store` is null. */
const render = async (store: ReturnType<typeof createWorkbenchStore> | null) => {
  workbench.store = store;
  stop = store ? start(store) : null;
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <AppToaster />
        <NotifyProbe />
      </ChakraProvider>
    )
  );
};

beforeEach(() => {
  focusMoves.length = 0;
});

afterEach(async () => {
  stop?.();
  await act(() => toaster.remove());
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  probe.notify = null;
  workbench.store = null;
});

const toastNamed = (title: string): HTMLElement | null =>
  [...document.querySelectorAll<HTMLElement>('[data-part="root"][data-scope="toast"]')].find(
    (toast) => toast.querySelector('[data-part="title"]')?.textContent === title
  ) ?? null;

const openToast = async (title: string): Promise<HTMLElement> => {
  await expect.poll(() => toastNamed(title)?.getAttribute('data-state')).toBe('open');
  return toastNamed(title)!;
};

const isGone = (title: string) => toastNamed(title)?.getAttribute('data-state') !== 'open';

const actionButtons = (toast: HTMLElement): HTMLButtonElement[] => [
  ...toast.querySelectorAll<HTMLButtonElement>('button:not([data-part="close-trigger"])'),
];

const buttonLabels = (toast: HTMLElement): string[] => actionButtons(toast).map((button) => button.textContent ?? '');

describe('notification toasts in the editor', () => {
  const store = () => workbench.store!;
  const recordedNotifications = () => store().getState().notifications;

  beforeEach(async () => {
    await render(createWorkbenchStore());
  });

  it('offers the actions this editor can run and dispatches the one pressed', async () => {
    await act(() =>
      probe.notify!.info('No image selected', 'Select an image in Gallery or Preview first.', {
        actions: [
          { kind: 'open-widget', widgetId: 'queue' },
          { kind: 'open-widget', widgetId: 'not-installed' },
        ],
      })
    );
    const toast = await openToast('No image selected');

    expect(buttonLabels(toast)).toEqual(['Open Queue']);
    await act(() => actionButtons(toast)[0]!.click());

    const project = store().getSnapshot().activeProject;
    expect(project.widgetInstances[project.widgetRegions.right.activeInstanceId]?.typeId).toBe('queue');
    expect(focusMoves).toEqual(['right:queue']);
    await expect.poll(() => isGone('No image selected')).toBe(true);
    // The record keeps both descriptors as data, so the center could still offer them.
    expect(structuredClone(recordedNotifications()[0]?.actions)).toEqual([
      { kind: 'open-widget', widgetId: 'queue' },
      { kind: 'open-widget', widgetId: 'not-installed' },
    ]);
  });

  it('undoes an undoable action once, from its toast, without a closure in workbench state', async () => {
    const onUndo = vi.fn();
    await act(() => probe.notify!.undoable('Moved 3 images', { message: 'To Portraits.', onUndo }));
    const toast = await openToast('Moved 3 images');
    const [recorded] = recordedNotifications();
    const undoAction = recorded?.actions?.[0];

    // Plain data: cloning would throw on a function.
    expect(structuredClone(recorded)).toEqual(recorded);
    expect(undoAction?.kind).toBe('undo');
    expect(buttonLabels(toast)).toEqual(['Undo']);

    await act(() => actionButtons(toast)[0]!.click());

    expect(onUndo).toHaveBeenCalledOnce();
    expect(undoAction?.kind === 'undo' && hasNotificationUndo(undoAction.undoId)).toBe(false);
  });

  it('forgets an undo whose toast was closed without it', async () => {
    const onUndo = vi.fn();
    await act(() => probe.notify!.undoable('Deleted board', { onUndo }));
    const toast = await openToast('Deleted board');
    const undoAction = recordedNotifications()[0]?.actions?.[0];

    await act(() => toast.querySelector<HTMLButtonElement>('[data-part="close-trigger"]')!.click());

    await expect.poll(() => undoAction?.kind === 'undo' && hasNotificationUndo(undoAction.undoId)).toBe(false);
    expect(onUndo).not.toHaveBeenCalled();
  });

  it('offers Undo only while the project it changed is open', async () => {
    const openProjectId = store().getSnapshot().activeProject.id;
    await act(() => {
      probe.notify!.undoable('Starred image', { onUndo: vi.fn(), projectId: openProjectId });
      probe.notify!.undoable('Starred elsewhere', { onUndo: vi.fn(), projectId: 'closed-project' });
    });

    expect(buttonLabels(await openToast('Starred image'))).toEqual(['Undo']);
    expect(buttonLabels(await openToast('Starred elsewhere'))).toEqual([]);
  });

  it('gives a persistent notice no timeout and leaves ordinary ones on the default', async () => {
    await act(() => {
      probe.notify!.error('Invocation failed', 'CUDA out of memory', { persistent: true });
      probe.notify!.success('Saved');
    });
    await openToast('Invocation failed');
    await openToast('Saved');
    const durationOf = (title: string) => toaster.getVisibleToasts().find((toast) => toast.title === title)?.duration;

    expect(durationOf('Invocation failed')).toBe(Number.POSITIVE_INFINITY);
    // Left to the toaster's per-type default.
    expect(durationOf('Saved')).toBeUndefined();
  });

  it('closes action and persistent toasts when the editor goes, and leaves ordinary notices', async () => {
    await act(() => {
      probe.notify!.info('No image selected', undefined, { actions: [{ kind: 'open-widget', widgetId: 'gallery' }] });
      probe.notify!.error('Invocation failed', 'CUDA out of memory', { persistent: true });
      probe.notify!.info('Workflow saved');
    });
    await openToast('No image selected');
    await openToast('Invocation failed');
    await openToast('Workflow saved');

    await act(() => stop?.());

    await expect.poll(() => isGone('No image selected')).toBe(true);
    await expect.poll(() => isGone('Invocation failed')).toBe(true);
    expect(toastNamed('Workflow saved')?.getAttribute('data-state')).toBe('open');
  });

  it('stays quiet about notifications recorded before it started', async () => {
    stop?.();
    await act(() => probe.notify!.info('Restored session notice'));
    stop = start(store());
    await act(() => probe.notify!.info('Fresh notice'));

    await openToast('Fresh notice');
    expect(toastNamed('Restored session notice')).toBeNull();
  });
});

describe('notification toasts outside the editor', () => {
  beforeEach(async () => {
    await render(null);
  });

  it('shows Undo straight on its toast and runs it once', async () => {
    const onUndo = vi.fn();
    await act(() => probe.notify!.undoable('Removed model', { onUndo }));
    const toast = await openToast('Removed model');

    expect(buttonLabels(toast)).toEqual(['Undo']);
    await act(() => actionButtons(toast)[0]!.click());

    expect(onUndo).toHaveBeenCalledOnce();
    await expect.poll(() => isGone('Removed model')).toBe(true);
  });

  it('closes persistent and undo toasts when the account changes', async () => {
    await act(() => {
      probe.notify!.error('Install failed', 'Disk full', { persistent: true });
      probe.notify!.undoable('Removed model', { onUndo: vi.fn() });
      probe.notify!.info('Model installed');
    });
    await openToast('Install failed');
    await openToast('Removed model');
    await openToast('Model installed');

    await act(() => accountLifecycle.invalidate());

    await expect.poll(() => isGone('Install failed')).toBe(true);
    await expect.poll(() => isGone('Removed model')).toBe(true);
    expect(toastNamed('Model installed')?.getAttribute('data-state')).toBe('open');
  });
});
