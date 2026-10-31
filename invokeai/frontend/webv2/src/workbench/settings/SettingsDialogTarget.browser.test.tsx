/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { Project } from '@workbench/projectContracts';
import type * as projectsApi from '@workbench/projects/api';
import type { WidgetRuntimeApi, WidgetViewProps } from '@workbench/widgetContracts';
import type * as workbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { canvasWidgetManifest } from '@workbench/widgets/canvas/manifest';
import { generateWidgetManifest } from '@workbench/widgets/generate/manifest';
import { getResolvedProjectWidgetValues, resolveWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { settingsDialogResource } from './dialogResource';
import { SettingsDialogHost } from './SettingsDialogHost';
import { closeWorkbenchSettings, settingsDialogStore } from './settingsDialogStore';
import { DEFAULT_PREFERENCES, patchWorkbenchPreferences } from './store';
import { WidgetSettingsButton } from './WidgetSettingsButton';

const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useActiveProjectId: () => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) => snapshot.activeProject.id);
  },
  useActiveProjectSelector: (selector: (project: Project) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) => selector(snapshot.activeProject));
  },
  useOptionalWorkbenchSelector: (selector: (snapshot: WorkbenchSnapshot) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, selector);
  },
  useWidgetInstanceValuesSelector: (instanceId: string, selector: (values: Record<string, unknown>) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) =>
      selector(resolveWidgetValues(snapshot.activeProject.widgetInstances[instanceId], snapshot.scopedDefaults))
    );
  },
  useWidgetValuesSelector: (widgetId: string, selector: (values: Record<string, unknown>) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) =>
      selector(getResolvedProjectWidgetValues(snapshot.activeProject, widgetId, snapshot.scopedDefaults))
    );
  },
  useOptionalWorkbenchCommands: () => storeRef.current!.commands,
  useOptionalWorkbenchQueries: () => storeRef.current!.queries,
  useWorkbenchCommands: () => storeRef.current!.commands,
  useWorkbenchQueries: () => storeRef.current!.queries,
  useWorkbenchSubscription: () => storeRef.current!.subscribe,
  useHasWorkbenchProvider: () => true,
}));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: translations } } });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let root: Root;
let host: HTMLDivElement;
let widgetProps: WidgetViewProps;

beforeEach(async () => {
  accountLifecycle.activate('settings-target-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  await settingsDialogResource.load();
  const state = createInitialWorkbenchState();
  const project = state.projects[0]!;
  const canvas = project.widgetInstances.canvas!;
  project.widgetInstances['canvas-copy'] = {
    ...canvas,
    id: 'canvas-copy',
    state: { ...canvas.state, values: { ...canvas.state.values, showGrid: true } },
  };
  storeRef.current = createWorkbenchStore(state);
  widgetProps = {
    instance: { ...project.widgetInstances['canvas-copy']!, title: 'Other Canvas' },
    manifest: canvasWidgetManifest,
    region: 'center',
    runtime: {} as WidgetRuntimeApi,
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  closeWorkbenchSettings();
  accountLifecycle.invalidate();
});

it('edits another widget through global search while retaining the originating Canvas instance', async () => {
  const store = storeRef.current!;
  const before = store.getSnapshot().activeProject;
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WidgetSettingsButton {...widgetProps} />
          <SettingsDialogHost />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  await act(() => page.getByRole('button', { name: 'Canvas settings', exact: true }).click());
  // The quick settings suspend on first open and then push the dialog button down. A click that starts on the button
  // during the fallback releases over a settings row instead, so it never activates the button.
  await expect
    .element(page.getByRole('checkbox', { name: i18n.t('widgets.canvas.settings.grid'), exact: true }))
    .toBeVisible();
  await act(() =>
    page
      .getByRole('button', { name: i18n.t('settingsDialog.allWidgetSettings', { widget: 'Canvas' }), exact: true })
      .click()
  );
  const search = page.getByRole('textbox', { name: i18n.t('settingsDialog.search'), exact: true });
  await expect.element(search).toBeVisible();
  await act(() => search.fill('filmstrip'));
  const filmstrip = page.getByRole('checkbox', { name: i18n.t('widgets.preview.showFilmstrip'), exact: true });
  await expect.element(filmstrip).toBeEnabled();
  await act(() => page.getByText(i18n.t('widgets.preview.showFilmstrip'), { exact: true }).click());
  await expect
    .poll(() => store.getSnapshot().activeProject.widgetInstances.preview!.state.values.filmstripVisible)
    .toBe(false);
  const afterPreview = store.getSnapshot().activeProject;
  expect(afterPreview.widgetInstances.canvas).toBe(before.widgetInstances.canvas);
  expect(afterPreview.widgetInstances['canvas-copy']).toBe(before.widgetInstances['canvas-copy']);
  expect(Object.keys(afterPreview.widgetInstances)).toEqual(Object.keys(before.widgetInstances));

  await act(() => search.fill('show grid'));
  const grid = page.getByRole('checkbox', { name: i18n.t('widgets.canvas.settings.grid'), exact: true });
  await expect.element(grid).toBeChecked();
  await act(() => page.getByText(i18n.t('widgets.canvas.settings.grid'), { exact: true }).click());
  await expect
    .poll(() => store.getSnapshot().activeProject.widgetInstances['canvas-copy']!.state.values.showGrid)
    .toBe(false);
  expect(store.getSnapshot().activeProject.widgetInstances.canvas).toBe(before.widgetInstances.canvas);
  expect(settingsDialogStore.getSnapshot().target).toEqual({ projectId: before.id, instanceId: 'canvas-copy' });
});

it('opens the pinned-setting menu from the gear popover and keeps focus inside the popover', async () => {
  const store = storeRef.current!;
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WidgetSettingsButton {...widgetProps} />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  await act(() => page.getByRole('button', { name: 'Canvas settings', exact: true }).click());
  const popoverContent = () => document.querySelector('[data-scope="popover"][data-part="content"]');
  const grid = page.getByRole('checkbox', { name: i18n.t('widgets.canvas.settings.grid'), exact: true });
  await expect.element(grid).toBeChecked();
  // The other Canvas pins its grid; only that row carries the dot.
  const dot = page.getByRole('button', {
    name: i18n.t('settingsDialog.layer.pinnedNamed', { setting: i18n.t('widgets.canvas.settings.grid') }),
    exact: true,
  });
  await expect.element(dot).toBeVisible();
  expect(dot.elements()).toHaveLength(1);

  await act(() => dot.click());
  const follow = page.getByRole('menuitem', { name: /Follow my default/ });
  await expect.element(follow).toBeVisible();
  await expect.element(page.getByText('Invoke default: Off', { exact: true })).toBeVisible();
  await userEvent.keyboard('{Escape}');
  await expect.element(follow).not.toBeInTheDocument();
  await expect.poll(() => popoverContent()?.contains(document.activeElement)).toBe(true);
  await expect.poll(() => document.activeElement).toBe(dot.element());

  await act(() => dot.click());
  await act(() => page.getByRole('menuitem', { name: /Follow my default/ }).click());
  await expect
    .poll(() =>
      Object.hasOwn(store.getSnapshot().activeProject.widgetInstances['canvas-copy']!.state.values, 'showGrid')
    )
    .toBe(false);
  await expect.element(grid).not.toBeChecked();
  await expect.poll(() => popoverContent()?.contains(document.activeElement)).toBe(true);
  await expect.poll(() => document.activeElement).toBe(grid.element());
});

it("shows the project's generation flags behind the Generate cog, and pins and follows them there", async () => {
  const store = storeRef.current!;
  const project = store.getSnapshot().activeProject;
  const generate = Object.values(project.widgetInstances).find((instance) => instance.typeId === 'generate')!;
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <WidgetSettingsButton
            {...widgetProps}
            instance={{ ...generate, title: 'Generate' }}
            manifest={generateWidgetManifest}
          />
        </I18nextProvider>
      </ChakraProvider>
    )
  );
  await act(() => page.getByRole('button', { name: 'Generate settings', exact: true }).click());
  for (const id of ['useCpuNoise', 'useNsfwChecker', 'useWatermark']) {
    await expect
      .element(page.getByRole('checkbox', { name: i18n.t(`settings.catalog.${id}.label`), exact: true }))
      .toBeVisible();
  }

  await act(() => page.getByText(i18n.t('settings.catalog.useNsfwChecker.label'), { exact: true }).click());
  // A project setting, not a value of the Generate widget.
  await expect.poll(() => store.getSnapshot().activeProject.settings.useNsfwChecker).toBe(true);
  expect(store.getSnapshot().activeProject.widgetInstances[generate.id]!.state.values).not.toHaveProperty(
    'useNsfwChecker'
  );

  await act(() =>
    page
      .getByRole('button', {
        name: i18n.t('settingsDialog.layer.pinnedNamed', { setting: i18n.t('settings.catalog.useNsfwChecker.label') }),
        exact: true,
      })
      .click()
  );
  await act(() => page.getByRole('menuitem', { name: 'Follow my default', exact: true }).click());
  await expect.poll(() => Object.hasOwn(store.getSnapshot().activeProject.settings, 'useNsfwChecker')).toBe(false);
  await expect
    .element(page.getByRole('checkbox', { name: i18n.t('settings.catalog.useNsfwChecker.label'), exact: true }))
    .not.toBeChecked();
});
