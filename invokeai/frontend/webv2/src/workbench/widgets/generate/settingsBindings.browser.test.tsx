/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { SettingDefinition, SettingsTarget } from '@platform/ui/settings/contracts';
import type * as projectsApi from '@workbench/projects/api';
import type { WorkbenchInternalStore, WorkbenchSnapshot } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { BUILT_IN_PROJECT_SETTINGS } from '@workbench/settings/projectSettings';
import { DEFAULT_PREFERENCES, getWorkbenchPreferences, patchWorkbenchPreferences } from '@workbench/settings/store';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { Field } from './settingsBindings';

const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useOptionalWorkbenchSelector: (selector: (snapshot: WorkbenchSnapshot) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, selector);
  },
  useOptionalWorkbenchCommands: () => storeRef.current!.commands,
  useOptionalWorkbenchQueries: () => storeRef.current!.queries,
}));
vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<typeof projectsApi>()),
  setClientStateValue: async () => {},
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: {} });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
const field: SettingDefinition = { id: 'useCpuNoise', kind: 'boolean', label: 'Use CPU noise', scope: 'project' };
let host: HTMLDivElement;
let root: Root;
const interact = (run: () => void) =>
  act(async () => {
    run();
    await Promise.resolve();
  });
const renderField = async (target?: SettingsTarget) => {
  await interact(() =>
    root.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Field field={field} surface="dialog" target={target} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return host.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
};

beforeEach(async () => {
  accountLifecycle.activate('application-settings-test');
  await patchWorkbenchPreferences(DEFAULT_PREFERENCES);
  storeRef.current = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await interact(() => root.unmount());
  host.remove();
});

describe('project settings bindings', () => {
  it('pins the current project, and edits the user default without a project target', async () => {
    const store = storeRef.current!;
    const target = { projectId: store.getSnapshot().activeProject.id };
    const control = await renderField(target);
    // A new project pins nothing: the control shows the built-in, and a click pins its opposite.
    expect(store.getSnapshot().activeProject.settings.useCpuNoise).toBeUndefined();
    expect(control.checked).toBe(BUILT_IN_PROJECT_SETTINGS.useCpuNoise);
    await interact(() => control.click());
    expect(store.getSnapshot().activeProject.settings.useCpuNoise).toBe(!BUILT_IN_PROJECT_SETTINGS.useCpuNoise);
    const defaults = await renderField();
    expect(defaults.disabled).toBe(false);
    expect(defaults.checked).toBe(BUILT_IN_PROJECT_SETTINGS.useCpuNoise);
    const before = store.getSnapshot();
    await interact(() => defaults.click());
    expect(getWorkbenchPreferences().scopedDefaults.generate).toEqual({
      useCpuNoise: !BUILT_IN_PROJECT_SETTINGS.useCpuNoise,
    });
    expect(store.getSnapshot()).toBe(before);
  });

  it('rejects a click delivered after project switching but before the old control commits its unmount', async () => {
    const store = storeRef.current!;
    const target = { projectId: store.getSnapshot().activeProject.id };
    const control = await renderField(target);
    await interact(() => {
      store.commands.projects.create();
      const beforeClick = store.getSnapshot();
      control.click();
      expect(store.getSnapshot()).toBe(beforeClick);
    });
    expect(control.disabled).toBe(true);
  });

  it('rejects a click from a control owned by an expired account session', async () => {
    const store = storeRef.current!;
    const control = await renderField({ projectId: store.getSnapshot().activeProject.id });
    await interact(() => {
      accountLifecycle.activate('next-account');
      const beforeClick = store.getSnapshot();
      control.click();
      expect(store.getSnapshot()).toBe(beforeClick);
    });
  });
});
