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
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

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
  await Promise.all([settingsDialogResource.load(), import('./WidgetQuickSettings')]);
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
