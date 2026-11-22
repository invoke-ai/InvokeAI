import type { ExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { WorkbenchFocusProvider } from '@workbench/WorkbenchRuntime';
import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

const runtime = vi.hoisted(() => ({
  /** Width of the header actions the fronted view contributes; wide ones crowd the start island to its floor. */
  actionsWidth: '0px',
  extensions: null as unknown as ExtensionRegistry,
  store: null as unknown as WorkbenchInternalStore,
}));
const useSnapshot = (): WorkbenchSnapshot => useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProjectId: () => useSnapshot().activeProject.id,
  useActiveProjectSelector: <Selected,>(selector: (project: Project) => Selected) =>
    selector(useSnapshot().activeProject),
  useOptionalWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchExtensions: () => runtime.extensions,
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchQueries: () => runtime.store.queries,
  useWorkbenchSelector: <Selected,>(selector: (snapshot: WorkbenchSnapshot) => Selected) => selector(useSnapshot()),
  useWorkbenchSubscription: () => runtime.store.subscribe,
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
}));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  setClientStateValue: () => Promise.resolve(),
}));
// The center's views stand in as a field and a button, inside the same hotkey-target markup the frame renders.
vi.mock('@workbench/widget-frame', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  WidgetChromeSlotById: ({ slot }: { slot: string }) =>
    slot === 'actions' ? <div style={{ flexShrink: 0, width: runtime.actionsWidth }} /> : null,
  WidgetRendererById: ({ instanceId, widget }: { instanceId: string; widget: { manifest: { id: string } } }) => (
    <div
      data-hotkey-widget-instance-id={instanceId}
      data-hotkey-widget-region="center"
      data-hotkey-widget-type-id={widget.manifest.id}
    >
      <input aria-label={`${instanceId} field`} />
      <button type="button">{instanceId} control</button>
    </div>
  ),
  WidgetSourceLockBadge: () => null,
}));

import { WorkbenchHotkeyRuntime } from '@workbench/hotkeys/WorkbenchHotkeyRuntime';

import { CenterArea } from './CenterArea';

const english = createInstance();
await english.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

let host: HTMLDivElement;
let root: Root;
let queryClient: QueryClient;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const centerFront = () => runtime.store.getSnapshot().activeProject.widgetRegions.center.activeInstanceId;
const centerRegion = () => host.querySelector<HTMLElement>('[data-focus-region="center"]')!;
const backButton = () => host.querySelector<HTMLButtonElement>('button[aria-label="Back to Canvas"]');
const press = (keys: string) => act(() => userEvent.keyboard(keys));
// The center's Escape runs once the press has reached every listener.
const afterDispatch = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 20);
      })
  );

const render = async (width: string) => {
  host.style.width = width;
  await act(() =>
    root.render(
      <I18nextProvider i18n={english}>
        <ChakraProvider value={system}>
          <QueryClientProvider client={queryClient}>
            <WorkbenchFocusProvider>
              <WorkbenchHotkeyRuntime />
              <CenterArea />
            </WorkbenchFocusProvider>
          </QueryClientProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

beforeEach(async () => {
  accountLifecycle.activate('back-to-stage-test');
  await patchWorkbenchPreferences({ customHotkeys: {} });
  runtime.store = createWorkbenchStore(
    workbenchReducer(createInitialWorkbenchState(), { presetId: 'edit', type: 'applyPreset' })
  );
  runtime.extensions = createExtensionRegistry();
  queryClient = new QueryClient();
  host = document.createElement('div');
  host.style.cssText = 'display:flex;height:480px;';
  document.body.append(host);
  root = createRoot(host);
  runtime.actionsWidth = '0px';
  const { activeProject } = runtime.store.getSnapshot();
  await act(() =>
    runtime.store.commands.widgets.select({ projectId: activeProject.id, region: 'center', widgetId: 'preview' })
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  queryClient?.clear();
  host?.remove();
  accountLifecycle.invalidate();
});

describe('Back to the stage', () => {
  it('leads the header while Away, fronts the stage and keeps focus in the center', async () => {
    await render('900px');
    const back = backButton();
    expect(back).not.toBeNull();
    expect(host.querySelector('[data-center-chrome-island] button')).toBe(back);

    await act(() => userEvent.click(back!));

    expect(centerFront()).toBe('canvas');
    expect(backButton()).toBeNull();
    await vi.waitFor(() => expect(centerRegion().contains(document.activeElement)).toBe(true));
  });

  it('stays visible and operable as an icon when wide header actions crowd a narrow center', async () => {
    runtime.actionsWidth = '600px';
    await render('340px');
    const back = backButton()!;
    const rect = back.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);

    expect(rect.width).toBeGreaterThanOrEqual(24);
    expect(rect.right).toBeLessThanOrEqual(centerRegion().getBoundingClientRect().right);
    expect(back.contains(hit)).toBe(true);
    expect(back.querySelector('p')!.getClientRects()).toHaveLength(0);

    await act(() => userEvent.click(back));
    expect(centerFront()).toBe('canvas');
  });

  it('takes Escape in the center only when nothing else handles it', async () => {
    await render('900px');
    await act(() => host.querySelector<HTMLInputElement>('[aria-label="preview field"]')!.focus());
    await press('{Escape}');
    await afterDispatch();
    expect(centerFront()).toBe('preview');

    const viewMenu = host.querySelector<HTMLButtonElement>('[aria-label="Center view: Preview"]')!;
    await act(() => userEvent.click(viewMenu));
    await vi.waitFor(() => expect(document.querySelector('[role="menu"]')).not.toBeNull());
    await press('{Escape}');
    await vi.waitFor(() => expect(viewMenu.getAttribute('aria-expanded')).toBe('false'));
    await afterDispatch();
    expect(centerFront()).toBe('preview');

    await act(() =>
      [...host.querySelectorAll('button')].find((button) => button.textContent === 'preview control')!.focus()
    );
    await press('{Escape}');
    await afterDispatch();
    expect(centerFront()).toBe('canvas');
  });
});
