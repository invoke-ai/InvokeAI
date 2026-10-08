/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasNodeContract } from '@workbench/canvas-engine/api';
import type { Project } from '@workbench/projectContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { closingFrames, recordDialogExit } from '@platform/ui/dialogExit.testing';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import { createDocumentModel } from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act, createRef, useImperativeHandle, useState, type Ref } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import type { LayerRowCommands } from './layerRowCommands';
import type * as RunLayerWorkflowDialogModule from './RunLayerWorkflowDialog';

import { LayerSurfaceHost, type LayerSurfaceEngine, type LayerSurfaceRequest } from './LayerSurfaceHost';

const PROJECT_ID = 'surface-project';
const mocks = vi.hoisted(() => ({
  commands: { notifications: { add: () => undefined }, widgets: { open: () => undefined } },
  // Only a layer menu reads workflow availability, once per render, so its calls show a layer menu is mounted.
  layerMenuRender: vi.fn(),
  project: null as unknown as Project,
}));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => PROJECT_ID,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(mocks.project),
  useOptionalWorkbenchCommands: () => mocks.commands,
  useWorkbenchCommands: () => mocks.commands,
}));
// A workflow with an image input and output, so the layer menu offers Run workflow.
vi.mock('./RunLayerWorkflowDialog', async (importOriginal) => {
  const availability = {
    canRunWorkflow: true,
    document: { id: 'workflow-1' },
    getRunnableInputs: () => [],
    hasWorkflowBindings: true,
    outputs: [],
  };
  return {
    ...(await importOriginal<typeof RunLayerWorkflowDialogModule>()),
    useLayerWorkflowAvailability: () => {
      mocks.layerMenuRender();
      return availability;
    },
  };
});

const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const nodes: CanvasNodeContract[] = [
  layerContract('layer', 'raster', { name: 'Layer' }),
  groupContract('group', [layerContract('child', 'raster', { name: 'Child' })], { name: 'Group' }),
];
const engine = {
  document: {
    model: () => createDocumentModel(mocks.project.canvas.document, { editRevision: 0, projectId: PROJECT_ID }),
  },
  exports: { hasExportableLayerContent: () => true },
  interaction: { get: () => false, subscribe: () => () => undefined },
  layers: { commitPrepared: () => ({ status: 'committed' as const }) },
  projectId: PROJECT_ID,
} as unknown as LayerSurfaceEngine;
const anchor = { height: 10, width: 10, x: 40, y: 40 };

const rowCommands = {} as LayerRowCommands;
type OpenSurface = (surface: LayerSurfaceRequest) => void;
const openSurfaceRef = createRef<OpenSurface>();
const openSurface: OpenSurface = (surface) => openSurfaceRef.current!(surface);
const Harness = ({ ref }: { ref: Ref<OpenSurface> }) => {
  const [surface, setSurface] = useState<LayerSurfaceRequest | null>(null);
  useImperativeHandle(ref, () => setSurface, []);
  return (
    <LayerSurfaceHost
      commands={rowCommands}
      dispatch={() => true}
      document={mocks.project.canvas.document}
      editingLocked={false}
      engine={engine}
      surface={surface}
      // The panel clears its surface as soon as the menu closes.
      onClose={() => setSurface(null)}
    />
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(async () => {
  const initial = { ...createInitialWorkbenchState().projects[0]!, id: PROJECT_ID };
  mocks.project = applyCanvasProjectMutation(initial, {
    document: { ...createEmptyCanvasDocument(), selectedLayerId: null, stacks: stacksFrom(nodes) },
    type: 'replaceCanvasDocument',
  });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <QueryClientProvider client={new QueryClient()}>
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Harness ref={openSurfaceRef} />
          </ChakraProvider>
        </I18nextProvider>
      </QueryClientProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it.each([
  ['layer', 'Rename layer', 'Rename layer'],
  ['layer', 'Run workflow', 'Run workflow from layer'],
  ['group', 'Rename layer', 'Rename layer'],
] as const)(
  'opens the %s menu "%s" dialog after the menu closes and animates it out on close',
  async (id, item, title) => {
    await act(() => openSurface({ anchor, id, kind: 'menu' }));
    await act(() => page.getByRole('menuitem', { name: item, exact: true }).click());

    await expect.element(page.getByRole('dialog', { name: title })).toBeVisible();
    const dialog = document.querySelector('[role="dialog"]')!;
    // Still open once the menu has gone: a dialog dismissed by the menu's teardown would already be closing.
    expect(dialog).toHaveAttribute('data-state', 'open');

    const frames = closingFrames(await recordDialogExit(dialog, () => act(() => userEvent.keyboard('{Escape}'))));

    expect(frames).not.toHaveLength(0);
    for (const frame of frames) {
      expect(frame.text).toContain(title);
    }
    await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();
  }
);

// A closed menu stays in the DOM until its own exit animation ends.
const openMenu = () => document.querySelector('[role="menu"][data-state="open"]');

/** Choose a menu's rename item, then hold its real exit animation while a successor opens. */
const renameThenDismiss = async (id: string) => {
  await act(() => openSurface({ anchor, id, kind: 'menu' }));
  await act(() => page.getByRole('menuitem', { name: 'Rename layer', exact: true }).click());
  await expect.element(page.getByRole('dialog', { name: 'Rename layer' })).toBeVisible();
  const dialog = document.querySelector<HTMLElement>('[role="dialog"]')!;
  await Promise.all(dialog.getAnimations().map((animation) => animation.finished));
  // Driver round trips can outlast the exit animation on CI; hold it before sending Escape.
  dialog.style.animationPlayState = 'paused';
  await act(() => userEvent.keyboard('{Escape}'));
  expect(dialog.isConnected).toBe(true);
  expect(dialog).toHaveAttribute('data-state', 'closed');
  const animations = dialog.getAnimations();
  expect(animations).not.toHaveLength(0);
  for (const animation of animations) {
    expect(animation.playState).toBe('paused');
  }
  return dialog;
};

it('drops a layer rename still animating out when another layer opens its menu', async () => {
  const dialog = await renameThenDismiss('layer');
  await act(() => openSurface({ anchor: { ...anchor }, id: 'child', kind: 'menu' }));
  await expect.poll(openMenu).not.toBeNull();
  expect(dialog.isConnected).toBe(false);

  mocks.layerMenuRender.mockClear();
  await act(() => userEvent.keyboard('{Escape}'));
  await expect.poll(() => document.querySelector('[role="menu"]')).toBeNull();

  // With every surface closed, no layer menu may come back for the rename its successor cut short.
  expect(mocks.layerMenuRender).not.toHaveBeenCalled();
});

it('opens a group menu requested while its rename animates out, and keeps it once that exit ends', async () => {
  const dialog = await renameThenDismiss('group');

  await act(() => openSurface({ anchor: { ...anchor }, id: 'group', kind: 'menu' }));
  await expect.poll(openMenu).not.toBeNull();
  expect(dialog.isConnected).toBe(true);
  dialog.style.removeProperty('animation-play-state');
  await expect.poll(() => dialog.isConnected).toBe(false);

  expect(openMenu()).not.toBeNull();
  await act(() => page.getByRole('menuitem', { name: 'Rename layer', exact: true }).click());
  await expect.element(page.getByRole('dialog', { name: 'Rename layer' })).toBeVisible();
});

it('starts each Run workflow open from a fresh form', async () => {
  const openRunWorkflow = async () => {
    await act(() => openSurface({ anchor, id: 'layer', kind: 'menu' }));
    await act(() => page.getByRole('menuitem', { name: 'Run workflow', exact: true }).click());
    await expect.element(page.getByRole('dialog', { name: 'Run workflow from layer' })).toBeVisible();
  };
  const destination = page.getByRole('combobox', { name: 'Destination' });

  await openRunWorkflow();
  await act(() => destination.click());
  await act(() => page.getByRole('option', { name: 'Canvas staging' }).click());
  await expect.element(destination).toHaveTextContent('Canvas staging');
  await act(() => userEvent.keyboard('{Escape}'));
  await expect.poll(() => document.querySelector('[role="dialog"]')).toBeNull();

  await openRunWorkflow();
  await expect.element(destination).toHaveTextContent('Gallery');
});
