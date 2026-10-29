/* oxlint-disable react-perf/jsx-no-new-object-as-prop */
import type { SettingsTarget } from '@platform/ui/settings/contracts';
import type { Project } from '@workbench/projectContracts';
import type { WorkbenchInternalStore } from '@workbench/workbenchStore';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { useExternalStoreSelector } from '@platform/state/selectors';
import { system } from '@theme/system';
import { Field as CanvasField } from '@workbench/widgets/canvas/settingsBindings';
import { canvasSettingsContribution } from '@workbench/widgets/canvas/settingsContribution';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useWidgetSettingsTarget } from './useWidgetSettingsTarget';

const storeRef = vi.hoisted(() => ({ current: null as WorkbenchInternalStore | null }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectSelector: (selector: (project: Project) => unknown) => {
    const store = storeRef.current!;
    return useExternalStoreSelector(store.subscribe, store.getSnapshot, (snapshot) => selector(snapshot.activeProject));
  },
  useWorkbenchCommands: () => storeRef.current!.commands,
  useWorkbenchQueries: () => storeRef.current!.queries,
}));

const i18n = createInstance();
await i18n.use(initReactI18next).init({ lng: 'en', resources: {} });
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let host: HTMLDivElement;
let root: Root;
const bindingRef = createRef<{ patch(values: Record<string, unknown>): void }>();
const Probe = ({ target }: { target?: SettingsTarget }) => {
  const binding = useWidgetSettingsTarget('canvas', target, (values) => values.showGrid === true);
  useImperativeHandle(bindingRef, () => binding, [binding]);
  return <output data-disabled={binding.disabled}>{String(binding.value)}</output>;
};

const interact = (run: () => void) =>
  act(async () => {
    run();
    await Promise.resolve();
  });

beforeEach(() => {
  accountLifecycle.activate('settings-test');
  storeRef.current = createWorkbenchStore();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});
afterEach(async () => {
  await interact(() => root.unmount());
  host.remove();
});

describe('widget settings target', () => {
  it('shares Canvas preferences between quick and dialog controls through the real aggregate', async () => {
    const field = canvasSettingsContribution.fields.find((candidate) => candidate.id === 'showGrid')!;
    await interact(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <CanvasField field={field} surface="quick" />
            <CanvasField field={field} surface="dialog" />
          </ChakraProvider>
        </I18nextProvider>
      )
    );
    const controls = Array.from(host.querySelectorAll<HTMLInputElement>('input[type="checkbox"]'));
    expect(controls).toHaveLength(2);
    expect(controls.every((control) => !control.checked)).toBe(true);
    await interact(() => controls[0]!.click());
    expect(controls.every((control) => control.checked)).toBe(true);
    expect(storeRef.current!.getSnapshot().activeProject.widgetInstances.canvas!.state.values.showGrid).toBe(true);
    await interact(() => controls[1]!.click());
    expect(controls.every((control) => !control.checked)).toBe(true);
  });

  it('edits an explicit instance without retargeting the default instance', async () => {
    const state = createInitialWorkbenchState();
    const project = state.projects[0]!;
    project.widgetInstances['canvas-copy'] = { ...project.widgetInstances.canvas!, id: 'canvas-copy' };
    storeRef.current = createWorkbenchStore(state);
    await interact(() => root.render(<Probe target={{ instanceId: 'canvas-copy', projectId: project.id }} />));
    await interact(() => bindingRef.current!.patch({ showGrid: true }));
    const instances = storeRef.current.getSnapshot().activeProject.widgetInstances;
    expect(instances['canvas-copy']!.state.values.showGrid).toBe(true);
    expect(instances.canvas!.state.values.showGrid).toBeUndefined();
  });

  it('disables a removed or wrong-type target without creating an instance', async () => {
    const store = storeRef.current!;
    const projectId = store.getSnapshot().activeProject.id;
    for (const instanceId of ['missing', 'preview']) {
      await interact(() => root.render(<Probe target={{ instanceId, projectId }} />));
      expect(host.querySelector('output')?.dataset.disabled).toBe('true');
      const before = store.getSnapshot();
      await interact(() => bindingRef.current!.patch({ showGrid: true }));
      expect(store.getSnapshot()).toBe(before);
    }
  });

  it('rejects stale writes after a project switch and account rotation', async () => {
    const store = storeRef.current!;
    await interact(() => root.render(<Probe />));
    const previousPatch = bindingRef.current!.patch;
    const firstProjectId = store.getSnapshot().activeProject.id;
    await interact(() => {
      store.commands.projects.create();
    });
    await interact(() => previousPatch({ showGrid: true }));
    expect(store.queries.getProject(firstProjectId)!.widgetInstances.canvas!.state.values.showGrid).toBeUndefined();
    accountLifecycle.activate('next-account');
    const before = store.getSnapshot();
    await interact(() => bindingRef.current!.patch({ showGrid: true }));
    expect(store.getSnapshot()).toBe(before);
  });
});
