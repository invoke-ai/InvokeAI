import type { WidgetRegion } from '@workbench/layoutContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Project, WorkbenchState } from '@workbench/projectContracts';
import type { RegisteredWidget, WidgetImplementation, WidgetViewProps } from '@workbench/widgetContracts';
import type * as WidgetRegistry from '@workbench/widgetRegistry';
import type * as WorkbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider, Portal } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { ConfirmDialog } from '@platform/ui/ConfirmDialog';
import { Dialog } from '@platform/ui/Dialog';
import { isModalPresent } from '@platform/ui/modalPresence';
import { toaster } from '@platform/ui/toaster';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { FloatingWidgetWindow } from '@workbench/widget-frame/FloatingWidgetWindow';
import { WidgetRendererById } from '@workbench/widget-frame/WidgetRenderer';
import { createWidgetImplementationResource } from '@workbench/widgetImplementationResource';
import { createInitialWorkbenchState, getPanelSizeBounds } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { createInstance } from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, useState, useSyncExternalStore, useCallback } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { CenterArea } from './CenterArea';
import { StatusBar } from './StatusBar';

const runtime = vi.hoisted(() => ({
  store: null as unknown as WorkbenchInternalStore,
  widgets: [] as RegisteredWidget[],
  reload: vi.fn(() => Promise.resolve()),
}));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const extensions = createExtensionRegistry();
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContext>()),
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchReload: () => runtime.reload,
  useWorkbenchCommands: () => runtime.store.commands,
  useOptionalWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchQueries: () => runtime.store.queries,
  useOptionalWorkbenchQueries: () => runtime.store.queries,
  useWorkbenchExtensions: () => extensions,
  useOptionalWorkbenchExtensions: () => extensions,
  useActiveProjectId: () =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => s.activeProject.id),
  useActiveProjectSelector: <T,>(select: (project: Project) => T, equal?: (a: T, b: T) => boolean) =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => select(s.activeProject), equal),
}));
vi.mock('@workbench/widgetRegistry', async (importOriginal) => ({
  ...(await importOriginal<typeof WidgetRegistry>()),
  getWidgetById: (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id),
  getWidgetsForRegion: (region: WidgetRegion) =>
    runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region)),
}));
const { registerWidgets } = await import('@workbench/widgetRegistry');
const getWidgetById = (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id);
const getWidgetsForRegion = (region: WidgetRegion) =>
  runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region));
const i18n = createInstance();
await i18n
  .use(initReactI18next)
  .init({ lng: 'en', resources: { en: { translation: await fetch('/locales/en.json').then((r) => r.json()) } } });
const Neighbor = () => {
  const [count, setCount] = useState(0);
  const increment = useCallback(() => setCount((value) => value + 1), []);
  return (
    <button type="button" onClick={increment}>
      Neighbor {count}
    </button>
  );
};
const DROP_STATE = { helperText: '', isActive: false, isAllowed: false };
const NEIGHBOR_PANEL_WIDTH_PX = getPanelSizeBounds('left').min;
type Host = 'center' | 'floating' | 'status' | 'dialog';
type Failure = 'lazy' | 'label' | 'actions' | 'body' | 'tooltip' | 'compact';
let host: HTMLDivElement;
let root: Root;
let mode: Host;
let broken: boolean;
let releaseLoad: (() => void) | null;
let attempts: number;
let failure: Failure;
const FixtureBody = ({ presentation }: WidgetViewProps) => {
  const [count, setCount] = useState(0);
  const increment = useCallback(() => setCount((value) => value + 1), []);
  if (
    broken &&
    (failure === 'body' ||
      (failure === 'tooltip' && presentation === 'tooltip') ||
      (failure === 'compact' && presentation === 'compact'))
  ) {
    throw new Error('body failed');
  }
  return presentation === 'compact' ? (
    <span>Fixture ready</span>
  ) : (
    <button type="button" style={{ marginTop: 64 }} onClick={increment}>
      Fixture ready {count}
    </button>
  );
};
const FixtureLabel = () => {
  if (broken && failure === 'label') {
    throw new Error('label failed');
  }
  return <span>Fixture label</span>;
};
const FixtureActions = () => {
  const [failed, setFailed] = useState(false);
  const fail = useCallback(() => setFailed(true), []);
  if (failed || (broken && failure === 'actions')) {
    throw new Error('actions failed');
  }
  return (
    <button type="button" onClick={fail}>
      Fixture action
    </button>
  );
};
const implementation: WidgetImplementation = {
  view: FixtureBody,
  headerLabel: FixtureLabel,
  headerActions: FixtureActions,
};
const load = () => {
  attempts += 1;
  if (failure === 'lazy' && attempts === 1) {
    return Promise.reject(new Error('chunk failed'));
  }
  if (releaseLoad) {
    return new Promise<WidgetImplementation>((resolve) => {
      releaseLoad = () => resolve(implementation);
    });
  }
  return Promise.resolve(implementation);
};
const ActualHost = () => {
  const snapshot = useSyncExternalStore(runtime.store.subscribe, runtime.store.getSnapshot);
  if (mode === 'center') {
    return <CenterArea />;
  }
  if (mode === 'dialog') {
    return (
      <Dialog.Root defaultOpen>
        <Portal>
          <Dialog.Backdrop />
          <Dialog.Positioner>
            <Dialog.Content>
              <Dialog.Header>
                <Dialog.Title>Fixture host</Dialog.Title>
              </Dialog.Header>
              <Dialog.Body>
                <WidgetRendererById instanceId="fixture" region="dialog" widget={runtime.widgets[0]!} />
              </Dialog.Body>
            </Dialog.Content>
          </Dialog.Positioner>
        </Portal>
      </Dialog.Root>
    );
  }
  if (mode === 'status') {
    return (
      <div style={{ display: 'flex', flex: 1, minWidth: 0, flexDirection: 'column' }}>
        <div style={{ display: 'flex', flex: 1, minHeight: 0, justifyContent: 'flex-end' }}>
          {failure === 'tooltip' ? (
            <WidgetRendererById instanceId="fixture" region="right" widget={runtime.widgets[0]!} />
          ) : null}
        </div>
        <StatusBar dropState={DROP_STATE} />
      </div>
    );
  }
  const window = snapshot.activeProject.floatingWidgets?.fixture;
  return window ? <FloatingWidgetWindow instanceId="fixture" stackRank={0} state={window} /> : null;
};
const makeState = (hostMode: Host): WorkbenchState => {
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  project.widgetInstances = Object.fromEntries(
    ['fixture', 'neighbor'].map((id) => [
      id,
      { id, typeId: id, createdAt: '2026-11-05', state: { id, label: id, values: {}, version: 1 } },
    ])
  );
  for (const region of Object.values(project.widgetRegions)) {
    region.instanceIds = [];
    region.activeInstanceId = '';
  }
  project.widgetRegions.left.sizePx = NEIGHBOR_PANEL_WIDTH_PX;
  project.widgetRegions.left.isCollapsed = false;
  project.widgetRegions.left.instanceIds = ['neighbor'];
  project.widgetRegions.left.activeInstanceId = 'neighbor';
  const region = project.widgetRegions[hostMode === 'center' ? 'center' : hostMode === 'status' ? 'bottom' : 'right'];
  region.instanceIds = ['fixture'];
  region.activeInstanceId = 'fixture';
  region.isCollapsed = false;
  return state;
};
const render = async (accountKey = 'recovery-hosts', extra?: React.ReactNode) => {
  const focus = createTestFocusController({
    getProjectId: () => runtime.store.getSnapshot().activeProject.id,
    isFloating: (id) => !!runtime.store.getSnapshot().activeProject.floatingWidgets?.[id],
  });
  await act(() =>
    root.render(
      <I18nextProvider key={accountKey} i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={focus}>
            <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
              <DndContext>
                {extra}
                <div style={{ width: NEIGHBOR_PANEL_WIDTH_PX, height: 150, flexShrink: 0 }}>
                  <WidgetRendererById instanceId="neighbor" region="left" widget={runtime.widgets[1]!} />
                </div>
                <div style={{ display: 'flex', flex: 1, minWidth: 0, height: '100%' }}>
                  <ActualHost />
                </div>
              </DndContext>
            </WorkbenchWidgetRegistryProvider>
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};
beforeEach(async () => {
  await page.viewport(1280, 900);
  accountLifecycle.activate('recovery-hosts', ':user:recovery-hosts');
  broken = true;
  runtime.reload.mockReset().mockResolvedValue(undefined);
  attempts = 0;
  releaseLoad = null;
  runtime.widgets = registerWidgets([
    {
      id: 'fixture',
      label: 'Fixture',
      version: 1,
      icon: CircleIcon,
      allowedRegions: ['left', 'right', 'bottom', 'center'],
      allowFloating: true,
      allowMultiple: false,
      bottomPanel: 'expandable',
      load,
      failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    },
    {
      id: 'neighbor',
      allowMultiple: false,
      label: 'Neighbor',
      version: 1,
      icon: CircleIcon,
      allowedRegions: ['left'],
      load: () => Promise.resolve({ view: Neighbor }),
      failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    },
  ]);
  host = document.createElement('div');
  host.style.cssText = 'display:flex;height:600px;width:1200px';
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});
describe('real shell recovery hosts', () => {
  it.each([
    ['center', 'lazy', 'expandable'],
    ['center', 'label', 'expandable'],
    ['center', 'actions', 'expandable'],
    ['center', 'body', 'expandable'],
    ['floating', 'lazy', 'expandable'],
    ['floating', 'actions', 'expandable'],
    ['floating', 'body', 'expandable'],
    ['status', 'lazy', 'expandable'],
    ['status', 'body', 'expandable'],
    ['status', 'lazy', 'tooltip'],
    ['status', 'body', 'tooltip'],
  ] as const)(
    '%s contains %s failure (%s) and retries once without replacing its neighbor',
    async (hostMode, failureMode, bottomPanel) => {
      mode = hostMode;
      failure = failureMode;
      runtime.widgets[0]!.manifest.bottomPanel = bottomPanel;
      runtime.store = createWorkbenchStore(makeState(mode));
      if (mode === 'floating') {
        runtime.store.commands.widgets.float('fixture', 'right');
        runtime.store.commands.widgets.setFloatingGeometry('fixture', { x: 500, y: 200, widthPx: 600, heightPx: 400 });
      }
      await render();
      await expect.poll(() => host.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      await userEvent.click(
        [...host.querySelectorAll('button')].find((button) => button.textContent === 'Neighbor 0')!
      );
      expect(host.textContent).toContain('Neighbor 1');
      if (mode === 'floating') {
        expect(host.querySelector('[data-floating-controls] button')).not.toBeNull();
      }
      if (mode === 'center') {
        expect(host.querySelector('[data-focus-region="center"]')?.getAttribute('aria-label')).toBe('Fixture');
      }
      const retry = [...host.querySelectorAll<HTMLButtonElement>('button')].find(
        (button) =>
          button.getAttribute('aria-label') === 'Fixture failed — select to retry' || button.textContent === 'Retry'
      )!;
      expect(retry.closest('button button, [role="button"] button')).toBeNull();
      if (mode === 'center' && (failureMode === 'body' || failureMode === 'lazy')) {
        const card = retry.parentElement!.parentElement!;
        const menu = host.querySelector<HTMLElement>('[aria-label="Center view: Fixture"]')!;
        expect(card.getBoundingClientRect().top).toBeGreaterThanOrEqual(menu.getBoundingClientRect().bottom);
      }
      const compactHost = retry.closest('[data-status-bar-widget]');
      if (mode === 'status' && failureMode === 'body') {
        for (let tab = 0; tab < 10 && document.activeElement !== retry; tab++) {
          await userEvent.keyboard('{Tab}');
        }
        expect(document.activeElement).toBe(retry);
        await expect.poll(() => document.querySelector('[role="tooltip"]')?.textContent).toContain('body failed');
        // Let the outer tooltip's delayed focus/hover activation run before checking the winner.
        await new Promise((resolve) => {
          setTimeout(resolve, 350);
        });
        const visibleTooltips = [...document.querySelectorAll<HTMLElement>('[role="tooltip"]')].filter(
          (tooltip) => tooltip.getBoundingClientRect().height > 0
        );
        expect(visibleTooltips).toHaveLength(1);
        expect(visibleTooltips[0]!.textContent).toBe('body failed');
        await expect
          .poll(() => {
            const bounds = document.querySelector('[role="tooltip"]')?.getBoundingClientRect();
            return !!bounds && bounds.height > 0 && bounds.top >= 0 && bounds.bottom <= window.innerHeight;
          })
          .toBe(true);
      }
      broken = false;
      if (mode === 'status' && failureMode === 'body') {
        await userEvent.keyboard('[Space]');
      } else {
        await userEvent.click(retry);
      }
      await expect.poll(() => host.querySelector('[data-testid="widget-failure"]')).toBeNull();
      await expect.poll(() => host.textContent).toContain('Fixture ready');
      expect(host.textContent).toContain('Neighbor 1');
      if (mode === 'status' && failureMode === 'body') {
        expect(document.activeElement).toBe(compactHost);
      }
      if (mode === 'status') {
        await expect
          .poll(() => host.querySelector('[data-status-bar-widget="fixture"] [role="status"]')?.textContent)
          .toBe('Fixture recovered');
      }
      expect(attempts).toBe(failureMode === 'lazy' ? 2 : 1);
    }
  );
  it.each(['center', 'floating'] as const)(
    'preserves the healthy %s body through a failed-again chrome retry',
    async (hostMode) => {
      mode = hostMode;
      failure = 'actions';
      runtime.store = createWorkbenchStore(makeState(mode));
      if (mode === 'floating') {
        runtime.store.commands.widgets.float('fixture', 'right');
      }
      await render();
      const body = [...host.querySelectorAll('button')].find((button) => button.textContent === 'Fixture ready 0')!;
      await userEvent.click(body);
      expect(body.textContent).toBe('Fixture ready 1');
      const retry = host.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — select to retry"]')!;
      await userEvent.click(retry);
      await expect
        .poll(() => host.querySelector('button[aria-label="Fixture failed — select to retry"]'))
        .not.toBeNull();
      expect([...host.querySelectorAll('button')].find((button) => button.textContent === 'Fixture ready 1')).toBe(
        body
      );
      expect(host.querySelectorAll('[data-testid="widget-failure"]')).toHaveLength(1);
    }
  );

  it('keeps an old placement failure out of current-host announcements', async () => {
    mode = 'status';
    failure = 'compact';
    runtime.store = createWorkbenchStore(makeState(mode));
    await render();
    await expect.poll(() => host.querySelector('button[aria-label="Fixture failed — select to retry"]')).not.toBeNull();
    await act(() => {
      mode = 'floating';
      runtime.store.commands.widgets.float('fixture', 'bottom');
      runtime.store.commands.widgets.setFloatingGeometry('fixture', { x: 500, y: 200, widthPx: 600, heightPx: 400 });
    });
    const window = host.querySelector<HTMLElement>('[data-floating-window="fixture"]')!;
    const body = [...window.querySelectorAll('button')].find((button) => button.textContent === 'Fixture ready 0')!;
    await userEvent.click(body);
    await userEvent.click(
      [...window.querySelectorAll('button')].find((button) => button.textContent === 'Fixture action')!
    );
    await expect.poll(() => window.querySelector('[role="status"]')?.textContent).toBe('Fixture failed');
    await userEvent.click(
      window.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — select to retry"]')!
    );
    await expect.poll(() => window.querySelector('[role="status"]')?.textContent).toBe('Fixture recovered');
    expect([...window.querySelectorAll('button')].find((button) => button.textContent === 'Fixture ready 1')).toBe(
      body
    );
  });

  it('contains a tooltip failure without replacing the healthy status row or expanded panel', async () => {
    mode = 'status';
    failure = 'tooltip';
    runtime.widgets[0]!.manifest.bottomPanel = 'tooltip';
    runtime.store = createWorkbenchStore(makeState(mode));
    await render();
    const row = host.querySelector<HTMLElement>('[data-status-bar-widget="fixture"] [role="button"]')!;
    await userEvent.hover(row);
    await expect.poll(() => document.querySelector('[role="tooltip"] [data-testid="widget-failure"]')).not.toBeNull();
    expect(host.querySelector('[data-status-bar-widget="fixture"] [role="button"]')).toBe(row);
    expect(row.textContent).toBe('Fixture ready');
    expect(host.querySelector('button[aria-label="Fixture failed — select to retry"]')).not.toBeNull();
    expect(host.querySelector('[data-testid="widget-failure"]')).toBeNull();
    expect(host.textContent).toContain('Fixture ready 0');
  });

  it('suppresses a healthy preserve-focus guide under an unrelated modal without reopening its popover', async () => {
    mode = 'status';
    failure = 'lazy';
    broken = false;
    Object.assign(runtime.widgets[0]!.manifest, {
      bottomPanel: 'popover',
      compactSizing: 'remaining-space',
      preserveWorkbenchFocus: true,
    });
    runtime.widgets[0]!.implementation = createWidgetImplementationResource('fixture', () =>
      Promise.resolve(implementation)
    );
    runtime.store = createWorkbenchStore(makeState(mode));
    const UnrelatedModal = () => {
      const [open, setOpen] = useState(false);
      const openModal = useCallback(() => setOpen(true), []);
      const close = useCallback(() => setOpen(false), []);
      return (
        <>
          <button type="button" data-open-unrelated onClick={openModal}>
            Open unrelated modal
          </button>
          <ConfirmDialog
            isOpen={open}
            title="Unrelated modal"
            body="Unrelated action"
            confirmLabel="Confirm"
            onClose={close}
            onConfirm={close}
          />
        </>
      );
    };
    await render(undefined, <UnrelatedModal />);
    const chip = page.getByRole('button', { name: 'Fixture ready', exact: true });
    (chip.element() as HTMLElement).focus();
    await userEvent.keyboard('{Enter}');
    await expect.element(page.getByRole('dialog')).toBeVisible();
    await act(() => host.querySelector<HTMLButtonElement>('[data-open-unrelated]')!.click());
    await expect.poll(() => isModalPresent()).toBe(true);
    await expect.element(page.getByRole('alertdialog', { name: 'Unrelated modal' })).toBeVisible();
    expect(host.querySelector('[data-status-bar-widget="fixture"]')).toBeNull();
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    await page.getByRole('alertdialog').getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect.element(chip).toBeVisible();
    await expect.poll(() => isModalPresent()).toBe(false);
    await expect.element(page.getByRole('dialog')).not.toBeInTheDocument();
    expect(runtime.reload).not.toHaveBeenCalled();
  });

  it.each(['status', 'floating', 'dialog'] as const)(
    'keeps confirmation consent and focus inside the %s host',
    async (hostMode) => {
      mode = hostMode;
      failure = 'lazy';
      runtime.widgets[0]!.manifest.bottomPanel = 'popover';
      runtime.widgets[0]!.implementation = createWidgetImplementationResource('fixture', () =>
        Promise.reject(new Error('repeat import failure'))
      );
      runtime.store = createWorkbenchStore(makeState(mode));
      if (mode === 'floating') {
        runtime.store.commands.widgets.float('fixture', 'right');
        runtime.store.commands.widgets.setFloatingGeometry('fixture', { x: 500, y: 200, widthPx: 600, heightPx: 400 });
      }
      await render();
      if (mode === 'status') {
        await page.getByRole('button', { name: 'Fixture failed — Open', exact: true }).click();
        await expect.element(page.getByRole('dialog')).toBeVisible();
      }
      const parentDialog = document.querySelector('[role="dialog"]');
      await page.getByRole('button', { name: 'Retry', exact: true }).click();
      const reload = page.getByRole('button', { name: 'Reload editor', exact: true });
      await reload.click();
      const confirm = page.getByRole('alertdialog', { name: 'Reload the editor?' });
      await expect.element(confirm).toBeVisible();
      await expect.poll(() => isModalPresent()).toBe(true);
      expect(runtime.reload).not.toHaveBeenCalled();
      await userEvent.keyboard('{Escape}');
      await expect.element(confirm).not.toBeInTheDocument();
      await expect.element(reload).toHaveFocus();
      if (parentDialog) {
        expect(parentDialog.isConnected).toBe(true);
      }
      expect(isModalPresent()).toBe(mode === 'dialog');
      await reload.click();
      await confirm.getByRole('button', { name: 'Cancel', exact: true }).click();
      await expect.element(reload).toHaveFocus();
      expect(runtime.reload).not.toHaveBeenCalled();
      expect(document.querySelector('button button, [role="button"] button')).toBeNull();
      await reload.click();
      await confirm.getByRole('button', { name: 'Reload editor', exact: true }).click();
      expect(runtime.reload).toHaveBeenCalledOnce();
      await expect.element(confirm).not.toBeInTheDocument();
      expect(isModalPresent()).toBe(mode === 'dialog');
      if (parentDialog) {
        expect(parentDialog.isConnected).toBe(true);
      }
    }
  );

  it('keeps a failed preserve-focus status item mounted through its own confirmation', async () => {
    mode = 'status';
    failure = 'lazy';
    Object.assign(runtime.widgets[0]!.manifest, {
      bottomPanel: 'popover',
      compactSizing: 'remaining-space',
      preserveWorkbenchFocus: true,
    });
    runtime.widgets[0]!.implementation = createWidgetImplementationResource('fixture', () =>
      Promise.reject(new Error('repeat import failure'))
    );
    runtime.store = createWorkbenchStore(makeState(mode));
    await render();
    await page.getByRole('button', { name: 'Fixture failed — select to retry', exact: true }).click();
    const reload = page.getByRole('button', { name: 'Fixture failed — Reload editor', exact: true });
    await reload.click();
    const confirm = page.getByRole('alertdialog', { name: 'Reload the editor?' });
    await expect.poll(() => isModalPresent()).toBe(true);
    await expect.element(confirm).toBeVisible();
    await expect.element(confirm.getByRole('button', { name: 'Cancel', exact: true })).toHaveFocus();
    expect(runtime.reload).not.toHaveBeenCalled();
    await confirm.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect.element(reload).toHaveFocus();
    await expect.poll(() => isModalPresent()).toBe(false);
    await reload.click();
    await expect.element(confirm).toBeVisible();
    await userEvent.keyboard('{Escape}');
    await expect.element(confirm).not.toBeInTheDocument();
    await expect.element(reload).toHaveFocus();
    expect(runtime.reload).not.toHaveBeenCalled();
    await reload.click();
    await confirm.getByRole('button', { name: 'Reload editor', exact: true }).click();
    expect(runtime.reload).toHaveBeenCalledOnce();
    await expect.poll(() => isModalPresent()).toBe(false);
  });

  it.each(['account', 'project', 'remove', 'unmount'] as const)(
    'keeps a pending reload refusal silent after %s departure',
    async (transition) => {
      mode = 'status';
      failure = 'lazy';
      const toast = vi.spyOn(toaster, 'create').mockImplementation(() => 'departed-reload');
      let reject!: (error: Error) => void;
      runtime.reload.mockImplementationOnce(
        () =>
          new Promise<void>((_resolve, rejectRun) => {
            reject = rejectRun;
          })
      );
      runtime.widgets[0]!.implementation = createWidgetImplementationResource('fixture', () =>
        Promise.reject(new Error('repeat import failure'))
      );
      runtime.store = createWorkbenchStore(makeState(mode));
      const project = runtime.store.getSnapshot().activeProject.id;
      await render();
      await page.getByRole('button', { name: 'Fixture failed — select to retry', exact: true }).click();
      await page.getByRole('button', { name: 'Fixture failed — Reload editor', exact: true }).click();
      await page.getByRole('alertdialog').getByRole('button', { name: 'Reload editor', exact: true }).click();
      expect(runtime.reload).toHaveBeenCalledOnce();
      if (transition === 'account') {
        accountLifecycle.activate('replacement', ':user:replacement');
      } else if (transition === 'project') {
        await act(() => {
          runtime.store.commands.projects.create();
        });
        expect(runtime.store.getSnapshot().activeProject.id).not.toBe(project);
      } else if (transition === 'remove') {
        await act(() => runtime.store.commands.widgets.toggle({ region: 'bottom', widgetId: 'fixture' }));
        expect(host.querySelector('[data-status-bar-widget="fixture"]')).toBeNull();
      } else {
        await act(() => root.unmount());
      }
      await act(async () => {
        reject(new Error('Late save refusal'));
        await Promise.resolve();
      });
      expect(runtime.reload).toHaveBeenCalledOnce();
      expect(toast).not.toHaveBeenCalled();
      toast.mockRestore();
    }
  );

  it.each(['center', 'status'] as const)(
    'cancels explicit %s reload without losing open projects’ edit histories',
    async (hostMode) => {
      mode = hostMode;
      failure = 'lazy';
      if (hostMode === 'status') {
        runtime.widgets[0]!.manifest.bottomPanel = 'popover';
      }
      runtime.widgets[0]!.implementation = createWidgetImplementationResource('fixture', () =>
        Promise.reject(new Error('deterministic import failure'))
      );
      runtime.store = createWorkbenchStore(makeState(mode));
      const first = runtime.store.getSnapshot().activeProject.id;
      runtime.store.commands.projects.create();
      for (const project of runtime.store.getSnapshot().projects) {
        runtime.store.commands.workflows.editGraph(
          { patch: { name: `Edited ${project.id}` }, type: 'setMetadata' },
          { projectId: project.id, workflowId: project.workflows.activeWorkflowId }
        );
      }
      runtime.store.commands.projects.switchTo(first);
      const histories = runtime.store.getSnapshot().projects.map((project) => project.workflowHistories);
      expect(histories.every((history) => Object.values(history).some((entry) => entry.past.length > 0))).toBe(true);
      await render();
      const selector =
        hostMode === 'status'
          ? 'button[aria-label="Fixture failed — select to retry"]'
          : '[data-testid="widget-failure"] button';
      await expect.poll(() => host.querySelector(selector)).not.toBeNull();
      const row = host.querySelector('[data-status-bar-widget="fixture"] [data-scope="popover"][data-part="trigger"]');
      if (hostMode === 'status') {
        expect(row).not.toBeNull();
      }
      await userEvent.click(host.querySelector<HTMLButtonElement>(selector)!);
      const reload =
        hostMode === 'status'
          ? page.getByRole('button', { name: 'Fixture failed — Reload editor', exact: true })
          : page.getByRole('button', { name: 'Reload editor', exact: true });
      await reload.click();
      const dialog = page.getByRole('alertdialog', { name: 'Reload the editor?' });
      await expect.element(dialog).toBeVisible();
      expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain('ALL open projects');
      await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
      await expect.element(reload).toHaveFocus();
      expect(runtime.reload).not.toHaveBeenCalled();
      expect(host.querySelector('button button')).toBeNull();
      expect(host.querySelector('[data-status-bar-widget="fixture"] [data-scope="popover"][data-part="trigger"]')).toBe(
        row
      );
      runtime.store.getSnapshot().projects.forEach((project, index) => {
        expect(project.workflowHistories).toBe(histories[index]);
        runtime.store.commands.workflows.undo({
          projectId: project.id,
          workflowId: project.workflows.activeWorkflowId,
        });
        expect(runtime.store.queries.getProject(project.id)?.workflows.entries[0]!.document.name).toBe(
          'Untitled Workflow'
        );
      });
    }
  );

  it('retires a tooltip failure when Retry succeeds with its tooltip closed', async () => {
    mode = 'status';
    failure = 'tooltip';
    runtime.widgets[0]!.manifest.bottomPanel = 'tooltip';
    runtime.store = createWorkbenchStore(makeState(mode));
    await render();
    const row = host.querySelector<HTMLElement>('[data-status-bar-widget="fixture"] [role="button"]')!;
    await userEvent.hover(row);
    await expect.poll(() => document.querySelector('[role="tooltip"] [data-testid="widget-failure"]')).not.toBeNull();
    await userEvent.unhover(row);
    await expect.poll(() => document.querySelector('[role="tooltip"] [data-testid="widget-failure"]')).toBeNull();
    broken = false;
    await userEvent.click(
      host.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — select to retry"]')!
    );
    await expect.poll(() => host.querySelector('button[aria-label="Fixture failed — select to retry"]')).toBeNull();
    await expect
      .poll(() => host.querySelector('[data-status-bar-widget="fixture"] [role="status"]')?.textContent)
      .toBe('Fixture recovered');
    expect(host.querySelector('[data-status-bar-widget="fixture"] [role="button"]')).toBe(row);
    expect(runtime.store.widgetRecovery.get('fixture')!.getSnapshot().errors).toEqual({});
  });

  it('retires an unmounted tooltip generation when a surviving panel retries the same instance', async () => {
    mode = 'status';
    failure = 'tooltip';
    runtime.widgets[0]!.manifest.bottomPanel = 'tooltip';
    const state = makeState(mode);
    state.projects[0]!.widgetRegions.right.instanceIds = ['fixture'];
    state.projects[0]!.widgetRegions.right.activeInstanceId = 'fixture';
    runtime.store = createWorkbenchStore(state);
    await render();
    const row = host.querySelector<HTMLElement>('[data-status-bar-widget="fixture"] [role="button"]')!;
    await userEvent.hover(row);
    await expect.poll(() => document.querySelector('[role="tooltip"] [data-testid="widget-failure"]')).not.toBeNull();
    await userEvent.unhover(row);
    await expect.poll(() => document.querySelector('[role="tooltip"] [data-testid="widget-failure"]')).toBeNull();
    await act(() => runtime.store.commands.widgets.toggle({ region: 'bottom', widgetId: 'fixture' }));
    expect(host.querySelector('[data-status-bar-widget="fixture"]')).toBeNull();
    const panel = host.querySelector<HTMLElement>('[data-focus-region="right"]')!;
    await userEvent.click(
      [...panel.querySelectorAll('button')].find((button) => button.textContent === 'Fixture action')!
    );
    broken = false;
    await userEvent.click(
      panel.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — select to retry"]')!
    );
    await expect.poll(() => runtime.store.widgetRecovery.get('fixture')!.getSnapshot().errors).toEqual({});
    await expect
      .poll(
        () =>
          host.querySelector('[data-focus-region="right"]')?.parentElement?.querySelector('[role="status"]')
            ?.textContent
      )
      .toBe('Fixture recovered');
    await act(() => runtime.store.commands.widgets.toggle({ region: 'bottom', widgetId: 'fixture' }));
    expect(host.querySelector('button[aria-label="Fixture failed — select to retry"]')).toBeNull();
  });

  it('keeps a stable status popover trigger and named focus host through failure and retry', async () => {
    mode = 'status';
    failure = 'body';
    runtime.widgets[0]!.manifest.bottomPanel = 'popover';
    runtime.store = createWorkbenchStore(makeState(mode));
    await render();
    const row = host.querySelector<HTMLElement>('[data-status-bar-widget="fixture"] [data-part="trigger"]')!;
    expect(row).not.toBeNull();
    await userEvent.click(host.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — Open"]')!);
    const popup = page.getByRole('dialog');
    await expect.element(popup).toBeVisible();
    const retry = popup.getByRole('button', { name: 'Retry', exact: true });
    await retry.click();
    await expect.element(retry).toBeVisible();
    expect(document.activeElement?.getAttribute('aria-label')).toBe('Fixture');
    expect(host.querySelector('[data-status-bar-widget="fixture"] [data-part="trigger"]')).toBe(row);
    await userEvent.keyboard('{Escape}');
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();
    await expect.poll(() => document.activeElement).toBe(host.querySelector('[data-status-bar-widget="fixture"]'));
    await userEvent.click(host.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — Open"]')!);
    await expect.element(popup).toBeVisible();
    broken = false;
    await retry.click();
    await expect.element(popup.getByRole('button', { name: 'Fixture ready 0', exact: true })).toBeVisible();
    expect(host.querySelector('[data-status-bar-widget="fixture"] [data-part="trigger"]')).toBe(row);
    await userEvent.keyboard('{Tab}');
    expect(document.activeElement?.textContent).toBe('Fixture ready 0');
  });

  it.each(['remove', 'account', 'unmount'] as const)(
    'does not revive a pending status recovery after %s',
    async (transition) => {
      mode = 'status';
      failure = 'lazy';
      runtime.store = createWorkbenchStore(makeState(mode));
      await render();
      await expect.poll(() => host.querySelector('[data-testid="widget-failure"]')).not.toBeNull();
      releaseLoad = () => {};
      await userEvent.click(
        host.querySelector<HTMLButtonElement>('button[aria-label="Fixture failed — select to retry"]')!
      );
      await expect.poll(() => attempts).toBe(2);
      if (transition === 'remove') {
        await act(() => runtime.store.commands.widgets.toggle({ region: 'bottom', widgetId: 'fixture' }));
      } else if (transition === 'account') {
        accountLifecycle.activate('replacement-account', ':user:replacement');
        runtime.store = createWorkbenchStore(makeState(mode));
        runtime.store.commands.widgets.toggle({ region: 'bottom', widgetId: 'fixture' });
        await render('replacement-account');
      } else {
        await act(() => root.unmount());
      }
      await act(async () => {
        releaseLoad?.();
        await Promise.resolve();
      });
      expect(host.textContent).not.toContain('Fixture ready');
      if (transition !== 'unmount') {
        expect(host.textContent).toContain('Neighbor 0');
      }
    }
  );
});
