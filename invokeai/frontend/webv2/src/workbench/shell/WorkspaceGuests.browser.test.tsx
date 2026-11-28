import type { WidgetRegion } from '@workbench/layoutContracts';
/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Project } from '@workbench/projectContracts';
import type { RegisteredWidget, WidgetManifest, WidgetViewProps } from '@workbench/widgetContracts';
import type * as WidgetRegistry from '@workbench/widgetRegistry';
import type * as WorkbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { createExtensionRegistry } from '@workbench/extensions/extensionRegistry';
import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { FloatingWidgetWindow } from '@workbench/widget-frame/FloatingWidgetWindow';
import { WidgetRendererById } from '@workbench/widget-frame/WidgetRenderer';
import { createInitialWorkbenchState, withArrangement } from '@workbench/workbenchState.testing';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { WorkbenchWidgetRegistryProvider } from '@workbench/WorkbenchWidgetRegistryContext';
import { getWorkspaceStructure } from '@workbench/workspace/accountWorkspaces';
import { openInWorkspace } from '@workbench/workspace/workspaceNavigation';
import { createInstance } from 'i18next';
import { CircleIcon } from 'lucide-react';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

const runtime = vi.hoisted(() => ({
  store: null as unknown as WorkbenchInternalStore,
  widgets: [] as RegisteredWidget[],
}));
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const extensions = createExtensionRegistry();

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContext>()),
  useWorkbenchInternalStore: () => runtime.store,
  useWorkbenchCommands: () => runtime.store.commands,
  useOptionalWorkbenchCommands: () => runtime.store.commands,
  useWorkbenchQueries: () => runtime.store.queries,
  useOptionalWorkbenchQueries: () => runtime.store.queries,
  useWorkbenchExtensions: () => extensions,
  useOptionalWorkbenchExtensions: () => extensions,
  useActiveProjectId: () =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => s.activeProject.id),
  useActiveLayoutSelector: <T,>(select: (layout: WorkbenchSnapshot['activeLayout']) => T) =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => select(s.activeLayout)),
  useActiveProjectSelector: <T,>(select: (project: Project) => T, equal?: (a: T, b: T) => boolean) =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, (s) => select(s.activeProject), equal),
  useWorkbenchSelector: <T,>(select: (snapshot: WorkbenchSnapshot) => T, equal?: (a: T, b: T) => boolean) =>
    useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, select, equal),
  // The lock reads Customize through this; without it the harness would never see a session.
  useOptionalWorkbenchSelector: <T,>(
    select: (snapshot: WorkbenchSnapshot) => T,
    _fallback: T,
    equal?: (a: T, b: T) => boolean
  ) => useExternalStoreSelector(runtime.store.subscribe, runtime.store.getSnapshot, select, equal),
}));
vi.mock('@workbench/widgetRegistry', async (importOriginal) => ({
  ...(await importOriginal<typeof WidgetRegistry>()),
  getWidgetById: (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id),
  getWidgetsForRegion: (region: WidgetRegion) =>
    runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region)),
}));

const { registerWidgets } = await import('@workbench/widgetRegistry');
const { CenterArea } = await import('./CenterArea');
const getWidgetById = (id: string) => runtime.widgets.find((widget) => widget.manifest.id === id);
const getWidgetsForRegion = (region: WidgetRegion) =>
  runtime.widgets.filter((widget) => widget.manifest.allowedRegions.includes(region));

const i18n = createInstance();
await i18n
  .use(initReactI18next)
  .init({ lng: 'en', resources: { en: { translation: await fetch('/locales/en.json').then((r) => r.json()) } } });

const FixtureView = ({ instance }: WidgetViewProps) => <p>{instance.typeId} body</p>;
const fixture = (id: string, label: string, manifest: Partial<WidgetManifest>): WidgetManifest => ({
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: CircleIcon,
  id,
  label,
  load: () => Promise.resolve({ view: FixtureView }),
  version: 1,
  ...manifest,
});

let host: HTMLDivElement;
let root: Root;

const layout = () => runtime.store.getSnapshot().activeLayout;
const project = () => runtime.store.getSnapshot().activeProject;
const button = (name: string) =>
  [...host.querySelectorAll<HTMLButtonElement>('button')].find(
    (candidate) => (candidate.getAttribute('aria-label') ?? candidate.textContent) === name
  );

const Windows = () => {
  const windows = useSyncExternalStore(runtime.store.subscribe, () => layout().floatingWidgets);

  return Object.entries(windows ?? {}).map(([instanceId, state]) => (
    <FloatingWidgetWindow key={instanceId} instanceId={instanceId} stackRank={0} state={state} />
  ));
};

const render = async () => {
  const focus = createTestFocusController({
    getProjectId: () => project().id,
    isFloating: (id) => !!layout().floatingWidgets?.[id],
  });
  const ports = { commands: runtime.store.commands, focus, getWidgetsForRegion, queries: runtime.store.queries };

  await act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <FocusRegionProvider controller={focus}>
            <WorkbenchWidgetRegistryProvider getWidgetById={getWidgetById} getWidgetsForRegion={getWidgetsForRegion}>
              <div style={{ display: 'flex', flex: 1, minWidth: 0 }}>
                <CenterArea />
              </div>
              <div style={{ display: 'flex', width: 360 }}>
                <WidgetRendererById instanceId="layers" region="right" widget={getWidgetById('layers')!} />
              </div>
              <Windows />
            </WorkbenchWidgetRegistryProvider>
          </FocusRegionProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );

  return ports;
};

beforeEach(async () => {
  await page.viewport(1280, 800);
  accountLifecycle.activate('workspace-guests', ':user:workspace-guests');
  runtime.widgets = registerWidgets([
    fixture('canvas', 'Canvas', { allowedRegions: ['center'] }),
    fixture('preview', 'Preview', { allowedRegions: ['center', 'right'] }),
    fixture('layers', 'Layers', { chrome: { body: 'fill', header: 'hidden' }, needs: ['canvas'] }),
    fixture('notes', 'Notes', {}),
    fixture('board', 'Board', { allowedRegions: ['center'] }),
  ]);
  runtime.store = createWorkbenchStore(
    withArrangement(createInitialWorkbenchState(), 'edit', {
      bottom: [],
      center: ['canvas', 'preview'],
      left: [],
      right: ['layers'],
    })
  );
  await act(() => runtime.store.commands.layout.applyPreset('edit'));
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

describe('workspace guests and needs in real frames', () => {
  it('says Canvas is hidden in the Layers frame while the center is Away, and its action brings Canvas back', async () => {
    await render();
    await expect.poll(() => host.textContent).toContain('layers body');
    expect(host.querySelector('[data-unmet-need]')).toBeNull();
    const layersBody = [...host.querySelectorAll('p')].find((element) => element.textContent === 'layers body')!;
    const topBefore = layersBody.getBoundingClientRect().top;

    await act(() => runtime.store.commands.widgets.select({ region: 'center', widgetId: 'preview' }));

    await expect
      .poll(() => host.querySelector('[data-unmet-need="canvas"]')?.textContent)
      .toContain('Canvas is hidden');
    // At the foot of the panel's body: what the user works with does not move.
    expect(layersBody.getBoundingClientRect().top).toBe(topBefore);
    expect(host.querySelector('[data-unmet-need]')!.getBoundingClientRect().top).toBeGreaterThan(topBefore);
    expect([...host.querySelectorAll('[role="status"]')].map(({ textContent }) => textContent)).toContain(
      'Canvas is hidden'
    );
    await userEvent.click(button('Show Canvas')!);

    expect(layout().widgetRegions.center.activeInstanceId).toBe('canvas');
    await expect.poll(() => host.querySelector('[data-unmet-need]')).toBeNull();
  });

  it.each([
    ['locked', false],
    ['rearranging without Customize', true],
  ])(
    'opens what no workspace places as a guest window that keeps, closes and forgets, but never docks, %s',
    async (_state, rearrangeWithoutCustomize) => {
      void patchWorkbenchPreferences({ rearrangeWithoutCustomize });
      const ports = await render();
      const hasDock = () =>
        [...host.querySelectorAll('button')].some((b) => b.getAttribute('aria-label')?.startsWith('Dock'));

      await act(() => openInWorkspace(ports, { typeId: 'notes' }));
      await expect.poll(() => button('Keep Notes in workspace')).toBeDefined();
      expect(hasDock()).toBe(false);

      await userEvent.click(button('Close Notes window')!);
      expect(layout().guests).toEqual({});
      expect(project().views.edit?.guests).toBeUndefined();
      await expect.poll(() => host.querySelector('[data-floating-window]')).toBeNull();

      await act(() => openInWorkspace(ports, { typeId: 'notes' }));
      await expect.poll(() => button('Keep Notes in workspace')).toBeDefined();
      await userEvent.click(button('Keep Notes in workspace')!);

      const session = runtime.store.getSnapshot().customizing?.[project().id];
      const kept = rearrangeWithoutCustomize
        ? getWorkspaceStructure(runtime.store.getSnapshot().account, 'edit')
        : session?.structure;
      expect(session === undefined).toBe(rearrangeWithoutCustomize);
      expect(Object.values(kept?.floatingWidgets ?? {})).toHaveLength(1);
      expect(layout().guests).toEqual({});
      await expect.poll(() => button('Keep Notes in workspace')).toBeUndefined();
      // Placed now, and structure is editable either way, so the same window docks.
      await expect.poll(hasDock).toBe(true);
    }
  );

  it('offers Keep, and no Close, for the fallback stage a center with nothing available shows', async () => {
    runtime.store = createWorkbenchStore(
      withArrangement(createInitialWorkbenchState(), 'edit', { center: ['retired:center'], right: ['layers'] })
    );
    await act(() => runtime.store.commands.layout.applyPreset('edit'));
    await render();
    await expect.poll(() => host.textContent).toContain('preview body');

    await userEvent.click(button('Center view: Preview')!);
    await expect.poll(() => document.querySelector('[data-value="keep-guest"]')).not.toBeNull();
    expect(document.querySelector('[data-value="close-guest"]')).toBeNull();
    await userEvent.click(document.querySelector<HTMLElement>('[data-value="keep-guest"]')!);

    const draft = runtime.store.getSnapshot().customizing?.[project().id]?.structure;
    expect(draft?.widgetRegions.center.instanceIds).toEqual(['retired:center', 'preview']);
    expect(layout().guests).toEqual({});
  });

  it('shows a center guest Away, with Back to the stage and Keep in its view menu', async () => {
    const ports = await render();

    await act(() => openInWorkspace(ports, { typeId: 'board' }));
    await expect.poll(() => host.textContent).toContain('board body');

    await userEvent.click(button('Center view: Board')!);
    await expect
      .poll(() => document.querySelector('[data-value="keep-guest"]')?.textContent)
      .toBe('Keep Board in workspace');
    await userEvent.keyboard('{Escape}');

    await userEvent.click(button('Back to Canvas')!);
    expect(layout().widgetRegions.center.activeInstanceId).toBe('canvas');
    expect(layout().guests).toEqual({});
  });
});
