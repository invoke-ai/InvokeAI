/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { SettingDefinition, SettingsTarget } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';
import type { UndoableNotice } from '@workbench/useNotify';
import type * as workbenchContext from '@workbench/WorkbenchContext';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import {
  hasNotificationUndo,
  registerNotificationUndo,
  runNotificationUndo,
} from '@workbench/shell/notifications/undoRegistry';
import { Field as CanvasField } from '@workbench/widgets/canvas/settingsBindings';
import { canvasSettingsContribution } from '@workbench/widgets/canvas/settingsContribution';
import { generateWidgetManifest } from '@workbench/widgets/generate/manifest';
import { Field as GenerateField } from '@workbench/widgets/generate/settingsBindings';
import { getResolvedProjectWidgetValues } from '@workbench/widgetState';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import {
  DEFAULT_PREFERENCES,
  getWorkbenchPreferences,
  patchWorkbenchPreferences,
  subscribeWorkbenchPreferences,
} from './store';
import { useLayeredSetting } from './useLayeredSetting';

/** The workbench the hooks see; null is the Launchpad, where there is none. */
const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));
const subscribeToNothing = () => () => {};
const getNull = () => null;
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof workbenchContext>()),
  useOptionalWorkbenchCommands: () => storeRef.current?.commands ?? null,
  useOptionalWorkbenchQueries: () => storeRef.current?.queries ?? null,
  useOptionalWorkbenchSelector: (selector: (snapshot: WorkbenchSnapshot) => unknown, fallback: unknown) => {
    const store = storeRef.current;
    return useExternalStoreSelector<WorkbenchSnapshot | null, unknown>(
      store?.subscribe ?? subscribeToNothing,
      store?.getSnapshot ?? getNull,
      (snapshot) => (snapshot ? selector(snapshot) : fallback)
    );
  },
}));
/** Undo notices as offered, each registered in the real registry so `supersedes` withdraws as it does in the app. */
const notices = vi.hoisted(() => [] as { title: string; notice: UndoableNotice; undoId: string }[]);
const notify = vi.hoisted(() => ({
  error: vi.fn(),
  info: vi.fn(),
  success: vi.fn(),
  undoable: (title: string, notice: UndoableNotice) => {
    notices.push({ notice, title, undoId: registerNotificationUndo(notice.onUndo, notice.supersedes) });
  },
}));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const translations = await fetch('/locales/en.json').then((response) => response.json());
const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: { en: { translation: translations } } });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const gridField = canvasSettingsContribution.fields.find((candidate) => candidate.id === 'showGrid')!;
const nsfwField = generateWidgetManifest.settings!.fields.find((candidate) => candidate.id === 'useNsfwChecker')!;
const gridDotName = (key: string) =>
  i18n.t(`settingsDialog.layer.${key}`, { setting: i18n.t('widgets.canvas.settings.grid') });
const readGrid = (values: Readonly<Record<string, unknown>>) => values.showGrid === true;

let host: HTMLDivElement;
let root: Root;
let stopMirror: () => void;
type Binding = ReturnType<typeof useLayeredSetting<boolean>>;
const bindingRef = createRef<Binding>();
const Probe = ({ field = gridField, target }: { field?: SettingDefinition; target?: SettingsTarget }) => {
  const binding = useLayeredSetting('canvas', 'showGrid', target, readGrid, false, field);
  useImperativeHandle(bindingRef, () => binding, [binding]);
  return <output data-disabled={binding.disabled}>{String(binding.value)}</output>;
};
const binding = () => bindingRef.current!;
const pinnedLayer = () => {
  const layer = binding().layer;
  if (layer?.state !== 'pinned') {
    throw new Error(`Expected a pinned layer, got ${layer?.state}`);
  }
  return layer;
};
const defaultLayer = () => {
  const layer = binding().layer;
  if (layer?.state !== 'default') {
    throw new Error(`Expected a default layer, got ${layer?.state}`);
  }
  return layer;
};

const renderWithProviders = (node: React.ReactNode) =>
  act(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{node}</ChakraProvider>
      </I18nextProvider>
    )
  );
const interact = (run: () => void) =>
  act(async () => {
    run();
    await Promise.resolve();
  });
/** Mounts a workbench and mirrors the settings store's defaults into it, as `WorkbenchRuntime` does. */
const mountWorkbench = (store = createWorkbenchStore()) => {
  storeRef.current = store;
  stopMirror = subscribeWorkbenchPreferences((preferences) =>
    store.internal.setScopedDefaults(preferences.scopedDefaults)
  );
  return store;
};
const canvasPins = (store: WorkbenchInternalStore, projectId = store.getSnapshot().activeProject.id) =>
  store.queries.getProject(projectId)!.widgetInstances.canvas!.state.values;

beforeEach(async () => {
  accountLifecycle.activate('layered-setting-test');
  notices.length = 0;
  notify.info.mockClear();
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  storeRef.current = null;
  stopMirror = () => {};
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await interact(() => root.unmount());
  host.remove();
  stopMirror();
  accountLifecycle.invalidate();
});

describe('layered settings inside a project', () => {
  it('pins quick and dialog controls of one instance without touching the user defaults', async () => {
    const store = mountWorkbench();
    const target = { projectId: store.getSnapshot().activeProject.id };
    await renderWithProviders(
      <>
        <CanvasField field={gridField} surface="quick" target={target} />
        <CanvasField field={gridField} surface="dialog" target={target} />
      </>
    );
    const controls = Array.from(host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
    expect(controls).toHaveLength(2);
    expect(controls.every((control) => !control.checked)).toBe(true);
    await interact(() => controls[0]!.click());
    expect(controls.every((control) => control.checked)).toBe(true);
    expect(canvasPins(store).showGrid).toBe(true);
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});
    // Pinned: the dot appears after the label, one per presentation.
    expect(page.getByRole('button', { name: gridDotName('pinnedNamed') }).elements()).toHaveLength(2);
  });

  it('shows a default from the Launchpad until the project pins its own value', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    expect(binding().value).toBe(true);
    expect(binding().layer).toBeUndefined();
    await interact(() => binding().set(false));
    expect(binding().value).toBe(false);
    expect(binding().layer).toMatchObject({ defaultValueLabel: 'Your default: On', state: 'pinned' });
    expect(getWorkbenchPreferences().scopedDefaults.canvas).toEqual({ showGrid: true });
  });

  it('makes a pin the default for every project without its own value, then follows it', async () => {
    const store = mountWorkbench();
    const firstId = store.getSnapshot().activeProject.id;
    await renderWithProviders(<Probe target={{ projectId: firstId }} />);
    await interact(() => binding().set(true));
    expect(pinnedLayer().defaultValueLabel).toBe('Invoke default: Off');

    await interact(() => pinnedLayer().onUseAsDefault());
    expect(getWorkbenchPreferences().scopedDefaults.canvas).toEqual({ showGrid: true });
    expect(Object.hasOwn(canvasPins(store, firstId), 'showGrid')).toBe(false);
    expect(binding().value).toBe(true);
    expect(binding().layer).toBeUndefined();

    await interact(() => {
      store.commands.projects.create();
    });
    const second = store.getSnapshot().activeProject;
    expect(second.id).not.toBe(firstId);
    expect(Object.hasOwn(second.widgetInstances.canvas!.state.values, 'showGrid')).toBe(false);
    expect(getResolvedProjectWidgetValues(second, 'canvas', store.getSnapshot().scopedDefaults).showGrid).toBe(true);
  });

  it('follows the default by deleting the pin', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(false));
    expect(canvasPins(store).showGrid).toBe(false);
    await interact(() => pinnedLayer().onFollowDefault());
    expect(Object.hasOwn(canvasPins(store), 'showGrid')).toBe(false);
    expect(binding().value).toBe(true);
    expect(binding().layer).toBeUndefined();
  });

  it('pins the project instance through an alias target and leaves the alias values untouched', async () => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    project.widgetInstances['canvas:alias'] = { ...project.widgetInstances.canvas!, id: 'canvas:alias' };
    const store = mountWorkbench(createWorkbenchStore(state));
    await renderWithProviders(<Probe target={{ instanceId: 'canvas:alias', projectId: project.id }} />);
    await interact(() => binding().set(true));
    const instances = store.getSnapshot().activeProject.widgetInstances;
    expect(instances.canvas!.state.values.showGrid).toBe(true);
    expect(Object.hasOwn(instances['canvas:alias']!.state.values, 'showGrid')).toBe(false);
    expect(pinnedLayer().state).toBe('pinned');
  });

  it('disables a removed or wrong-type target without creating an instance', async () => {
    const store = mountWorkbench();
    const projectId = store.getSnapshot().activeProject.id;
    for (const instanceId of ['missing', 'preview']) {
      await renderWithProviders(<Probe target={{ instanceId, projectId }} />);
      expect(host.querySelector('output')?.dataset.disabled).toBe('true');
      const before = store.getSnapshot();
      await interact(() => binding().set(true));
      expect(store.getSnapshot()).toBe(before);
    }
  });

  it('rejects stale writes after a project switch and account rotation', async () => {
    const store = mountWorkbench();
    const firstProjectId = store.getSnapshot().activeProject.id;
    await renderWithProviders(<Probe target={{ projectId: firstProjectId }} />);
    const previousSet = binding().set;
    await interact(() => {
      store.commands.projects.create();
    });
    expect(host.querySelector('output')?.dataset.disabled).toBe('true');
    await interact(() => previousSet(true));
    expect(canvasPins(store, firstProjectId).showGrid).toBeUndefined();
    accountLifecycle.activate('next-account');
    const before = store.getSnapshot();
    await interact(() => binding().set(true));
    expect(store.getSnapshot()).toBe(before);
  });

  it('pins the generation flags on the project settings and follows the default again', async () => {
    const store = mountWorkbench();
    const target = { projectId: store.getSnapshot().activeProject.id };
    await renderWithProviders(<GenerateField field={nsfwField} surface="dialog" target={target} />);
    const toggle = page.getByRole('checkbox', { name: i18n.t('settings.catalog.useNsfwChecker.label'), exact: true });
    await expect.element(toggle).not.toBeChecked();
    await act(() => page.getByText(i18n.t('settings.catalog.useNsfwChecker.label'), { exact: true }).click());
    expect(store.getSnapshot().activeProject.settings.useNsfwChecker).toBe(true);
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});

    await act(() =>
      page
        .getByRole('button', { name: i18n.t('settingsDialog.layer.pinnedNamed', { setting: 'NSFW checker' }) })
        .click()
    );
    await expect.element(page.getByText('Invoke default: Off')).toBeVisible();
    await act(() => page.getByRole('menuitem', { name: /Follow my default/ }).click());
    expect(Object.hasOwn(store.getSnapshot().activeProject.settings, 'useNsfwChecker')).toBe(false);
    await expect.element(toggle).not.toBeChecked();
  });
});

describe('undoing a dot action', () => {
  const lastNotice = () => notices.at(-1)!;
  const undo = (entry = lastNotice()) => interact(() => runNotificationUndo(entry.undoId));
  const defaults = () => getWorkbenchPreferences().scopedDefaults;

  it('undoes Use as my default by restoring the earlier default and the pin', async () => {
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onUseAsDefault());
    expect(lastNotice().title).toBe('Your default for Show grid is now On');
    expect(defaults()).toEqual({ canvas: { showGrid: true } });
    await undo();
    expect(defaults()).toEqual({});
    expect(canvasPins(store).showGrid).toBe(true);

    // A pin equal to the built-in removes the default it replaces; Undo brings that default back.
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onUseAsDefault());
    expect(lastNotice().title).toBe('Show grid follows the Invoke default again (Off)');
    expect(defaults()).toEqual({});
    await undo();
    expect(defaults()).toEqual({ canvas: { showGrid: true } });
    expect(canvasPins(store).showGrid).toBe(false);
  });

  it('undoes Follow my default by pinning the old value again', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onFollowDefault());
    expect(lastNotice().title).toBe('Show grid follows your default (On)');
    expect(lastNotice().notice.projectId).toBe(store.getSnapshot().activeProject.id);
    await undo();
    expect(canvasPins(store).showGrid).toBe(false);
    expect(defaults()).toEqual({ canvas: { showGrid: true } });
  });

  it('undoes Reset to Invoke default by restoring the default, account-wide', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    await renderWithProviders(<Probe />);
    await interact(() => defaultLayer().onResetToBuiltIn!());
    expect(defaults()).toEqual({});
    expect(lastNotice().notice).not.toHaveProperty('projectId');
    await undo();
    expect(defaults()).toEqual({ canvas: { showGrid: true } });
  });

  it('keeps an edit made since the action', async () => {
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onUseAsDefault());
    await patchWorkbenchPreferences({ scopedDefaults: {} });
    await undo();
    expect(defaults()).toEqual({});
    expect(Object.hasOwn(canvasPins(store), 'showGrid')).toBe(false);
    expect(notify.info).toHaveBeenCalledWith('Nothing to undo: Show grid changed since.');
  });

  it('keeps a different pin made since following the default', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onFollowDefault());
    await interact(() => binding().set(true));
    await undo();
    expect(canvasPins(store).showGrid).toBe(true);
    expect(notify.info).toHaveBeenCalledWith('Nothing to undo: Show grid changed since.');
  });

  it('keeps a default set since a pin at the built-in removed it', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onUseAsDefault());
    expect(defaults()).toEqual({});
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: false } } });
    await undo();
    expect(defaults()).toEqual({ canvas: { showGrid: false } });
    expect(Object.hasOwn(canvasPins(store), 'showGrid')).toBe(false);
    expect(notify.info).toHaveBeenCalledWith('Nothing to undo: Show grid changed since.');
  });

  it('restores a pin in a project that is open but no longer active, each project keeping its own Undo', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    const firstId = store.getSnapshot().activeProject.id;
    await renderWithProviders(<Probe target={{ projectId: firstId }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onFollowDefault());
    const first = lastNotice();

    let secondId = '';
    await act(() => {
      secondId = store.commands.projects.create().id;
    });
    await renderWithProviders(<Probe target={{ projectId: secondId }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onFollowDefault());
    expect(hasNotificationUndo(first.undoId)).toBe(true);

    await undo(first);
    expect(canvasPins(store, firstId).showGrid).toBe(false);
    expect(Object.hasOwn(canvasPins(store, secondId), 'showGrid')).toBe(false);
    expect(store.getSnapshot().activeProject.id).toBe(secondId);
  });

  it('restores through the project instance after the alias the action came from is gone', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    project.widgetInstances['canvas:alias'] = { ...project.widgetInstances.canvas!, id: 'canvas:alias' };
    const store = mountWorkbench(createWorkbenchStore(state));
    await renderWithProviders(<Probe target={{ instanceId: 'canvas:alias', projectId: project.id }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onFollowDefault());

    const current = store.getState();
    const { 'canvas:alias': _removed, ...widgetInstances } = current.projects[0]!.widgetInstances;
    await interact(() =>
      store.internal.persistence.hydrate({ ...current, projects: [{ ...current.projects[0]!, widgetInstances }] })
    );
    expect(store.getSnapshot().activeProject.widgetInstances['canvas:alias']).toBeUndefined();
    await undo();
    expect(canvasPins(store).showGrid).toBe(false);
  });

  it('restores a project setting in its project while another project is active', async () => {
    const store = mountWorkbench();
    const firstId = store.getSnapshot().activeProject.id;
    const nsfwRead = (values: Readonly<Record<string, unknown>>) => values.useNsfwChecker === true;
    const NsfwProbe = ({ target }: { target: SettingsTarget }) => {
      const nsfw = useLayeredSetting('generate', 'useNsfwChecker', target, nsfwRead, false, nsfwField);
      useImperativeHandle(bindingRef, () => nsfw, [nsfw]);
      return null;
    };
    await renderWithProviders(<NsfwProbe target={{ projectId: firstId }} />);
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onFollowDefault());
    expect(Object.hasOwn(store.getSnapshot().activeProject.settings, 'useNsfwChecker')).toBe(false);

    let secondId = '';
    await act(() => {
      secondId = store.commands.projects.create().id;
    });
    await undo();
    expect(store.queries.getProject(firstId)!.settings.useNsfwChecker).toBe(true);
    expect(Object.hasOwn(store.queries.getProject(secondId)!.settings, 'useNsfwChecker')).toBe(false);
    expect(store.getSnapshot().activeProject.id).toBe(secondId);
  });

  it('withdraws the earlier Undo when a later action changes the same default or pin', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    const store = mountWorkbench();
    await renderWithProviders(<Probe target={{ projectId: store.getSnapshot().activeProject.id }} />);
    await interact(() => binding().set(false));
    await interact(() => pinnedLayer().onUseAsDefault());
    const first = lastNotice();
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onUseAsDefault());
    expect(hasNotificationUndo(first.undoId)).toBe(false);
    await undo();
    expect(defaults()).toEqual({});
    expect(canvasPins(store).showGrid).toBe(true);

    // A Follow Undo would re-pin over the default Use as my default sets, so that action withdraws it.
    await interact(() => pinnedLayer().onFollowDefault());
    const follow = lastNotice();
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onUseAsDefault());
    const useAsDefault = lastNotice();
    expect(defaults()).toEqual({ canvas: { showGrid: true } });
    expect(hasNotificationUndo(follow.undoId)).toBe(false);
    await undo(follow);
    expect(Object.hasOwn(canvasPins(store), 'showGrid')).toBe(false);
    await undo(useAsDefault);
    expect(defaults()).toEqual({});
    expect(canvasPins(store).showGrid).toBe(true);
    expect(notify.info).not.toHaveBeenCalled();

    // Following the default in the same project twice: only the latest Undo stays.
    await interact(() => pinnedLayer().onFollowDefault());
    const firstFollow = lastNotice();
    await interact(() => binding().set(true));
    await interact(() => pinnedLayer().onFollowDefault());
    expect(hasNotificationUndo(firstFollow.undoId)).toBe(false);
  });

  it('does nothing once the account has changed', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    await renderWithProviders(<Probe />);
    await interact(() => defaultLayer().onResetToBuiltIn!());
    const { notice } = lastNotice();
    accountLifecycle.activate('next-account');
    const before = getWorkbenchPreferences();
    await interact(() => notice.onUndo());
    expect(getWorkbenchPreferences()).toBe(before);
    expect(notify.info).not.toHaveBeenCalled();
  });
});

describe('layered settings in the Launchpad', () => {
  it('edits the user default, drops it at the built-in, and resets to the built-in', async () => {
    await renderWithProviders(<Probe />);
    expect(binding()).toMatchObject({ disabled: false, layer: undefined, value: false });
    await interact(() => binding().set(true));
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({ canvas: { showGrid: true } });
    expect(binding().layer?.state).toBe('default');
    await interact(() => binding().set(false));
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});

    await interact(() => binding().set(true));
    await interact(() => defaultLayer().onResetToBuiltIn!());
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});
    expect(binding().value).toBe(false);
  });

  it('resets from the keyboard and leaves focus on the row control once the dot is gone', async () => {
    await patchWorkbenchPreferences({ scopedDefaults: { canvas: { showGrid: true } } });
    await renderWithProviders(
      <>
        <button type="button">Before</button>
        <CanvasField field={gridField} surface="dialog" />
      </>
    );
    const toggle = page.getByRole('checkbox', { name: i18n.t('widgets.canvas.settings.grid'), exact: true });
    await expect.element(toggle).toBeChecked();
    const dot = page.getByRole('button', { name: gridDotName('changedNamed'), exact: true });
    (page.getByRole('button', { name: 'Before', exact: true }).element() as HTMLElement).focus();
    // The dot follows the label, ahead of the switch in tab order.
    await userEvent.keyboard('{Tab}');
    await expect.poll(() => document.activeElement).toBe(dot.element());
    await userEvent.keyboard('{Enter}');
    const reset = page.getByRole('menuitem', { name: i18n.t('settingsDialog.layer.resetToBuiltIn') });
    await expect.element(reset).toBeVisible();
    await userEvent.keyboard('{ArrowDown}{Enter}');
    await expect.element(toggle).not.toBeChecked();
    expect(getWorkbenchPreferences().scopedDefaults).toEqual({});
    await expect.element(reset).not.toBeInTheDocument();
    await expect.element(dot).not.toBeInTheDocument();
    await expect.poll(() => document.activeElement).toBe(toggle.element());
  });
});
