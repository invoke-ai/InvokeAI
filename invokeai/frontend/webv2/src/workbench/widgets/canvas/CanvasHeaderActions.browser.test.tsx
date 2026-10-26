import type { CanvasLayerContract } from '@workbench/canvas-engine/contracts';
import type { CanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import type { Project } from '@workbench/projectContracts';
import type { WidgetViewProps } from '@workbench/widgetContracts';
import type * as WorkbenchContextModule from '@workbench/WorkbenchContext';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { stacksFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

import type * as EngineStoreHooks from './engineStoreHooks';

/** The actions read the real store and drive a real engine; only the editing lock and gallery save are stubbed. */
const harness = vi.hoisted(() => ({
  editingLocked: false,
  engine: null as CanvasEngine | null,
  store: null as ReturnType<typeof createWorkbenchStore> | null,
}));
const activeProject = () => harness.store!.getState().projects[0]!;

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<typeof WorkbenchContextModule>()),
  useActiveProjectId: () => activeProject().id,
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(activeProject()),
  useOptionalWorkbenchCommands: () => harness.store!.commands,
  useWorkbenchCommands: () => harness.store!.commands,
}));
vi.mock('./engineStoreHooks', async (importOriginal) => ({
  ...(await importOriginal<typeof EngineStoreHooks>()),
  useCanvasDocumentEditingLocked: () => harness.editingLocked,
}));
vi.mock('@workbench/canvas-operations/react', () => ({ useCanvasEngine: () => harness.engine }));
vi.mock('./useCanvasGallerySave', () => ({
  useCanvasGallerySave: () => ({ isSaving: false, save: () => Promise.resolve() }),
}));

import { CanvasHeaderActions } from './CanvasHeaderActions';

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  harness.engine?.lifecycle.dispose();
  Object.assign(harness, { editingLocked: false, engine: null, store: null });
  root = null;
  host = null;
});

interface Registration {
  defaultKeys?: string[];
  handler?: () => void;
  id: string;
  title: string;
}

/** A painted layer, so a cleared document differs from the one it replaced. */
const paintedLayer: CanvasLayerContract = {
  blendMode: 'normal' as const,
  id: 'painted',
  isEnabled: true,
  isLocked: false,
  name: 'Painted',
  opacity: 1,
  source: { image: { height: 1, imageName: 'painted.png', width: 1 }, type: 'image' as const },
  transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 0, y: 0 },
  type: 'raster' as const,
};

/**
 * Renders the actions on a fresh project holding `layers` (none by default) and the canvas widget values
 * `canvasValues`, recording the commands and hotkeys they register.
 */
const renderActions = async ({
  canvasValues,
  layers = [],
}: { canvasValues?: Record<string, unknown>; layers?: CanvasLayerContract[] } = {}) => {
  const initial = createInitialWorkbenchState();
  const base = initial.projects[0]!;
  const store = createWorkbenchStore(
    layers.length > 0
      ? {
          ...initial,
          projects: [
            {
              ...base,
              canvas: {
                ...base.canvas,
                document: { ...base.canvas.document, stacks: stacksFrom(layers) },
              },
            },
          ],
        }
      : initial
  );
  if (canvasValues) {
    store.commands.widgets.patchValues('canvas', canvasValues);
  }
  const projectId = store.getState().activeProjectId;
  harness.store = store;
  harness.engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    ensureProjectOnServer: () => Promise.resolve(),
    imageResolver: () => Promise.resolve(new Blob()),
    mutationPort: createCanvasProjectMutationPort(store, projectId),
    projectId,
    reportError: () => undefined,
  });
  const commands: Registration[] = [];
  const hotkeys: Registration[] = [];
  const runtime = {
    commands: {
      register: (command: Registration) => {
        commands.push(command);
        return () => undefined;
      },
    },
    hotkeys: {
      register: (hotkey: Registration) => {
        hotkeys.push(hotkey);
        return () => undefined;
      },
    },
  };
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <CanvasHeaderActions {...({ runtime } as unknown as WidgetViewProps)} />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  return { commands, hotkeys, store };
};

const clearButton = () => page.getByRole('button', { exact: true, name: 'Clear canvas' });
const confirmDialog = () => page.getByRole('alertdialog');

describe('CanvasHeaderActions clear canvas', () => {
  it('asks to clear the canvas on one click, without a menu, even when the canvas is empty', async () => {
    await renderActions();

    await clearButton().click();

    await expect.element(confirmDialog()).toBeVisible();
    await expect.element(confirmDialog().getByRole('heading', { name: 'Clear canvas' })).toBeVisible();
    await expect.element(confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' })).toBeVisible();
    await expect
      .element(confirmDialog().getByText('Clear all layers and start a fresh canvas? This cannot be undone.'))
      .toBeVisible();
    expect(document.querySelector('[role="menu"]')).toBeNull();
  });

  it('replaces the document once confirmed, layers and all', async () => {
    const { store } = await renderActions({ layers: [paintedLayer] });
    const before = store.getState().projects[0]!.canvas.documentRevision;
    expect(store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(1);

    await clearButton().click();
    await confirmDialog().getByRole('button', { exact: true, name: 'Clear canvas' }).click();

    await expect.element(confirmDialog()).not.toBeInTheDocument();
    expect(store.getState().projects[0]!.canvas.documentRevision).toBeGreaterThan(before);
    expect(store.getState().projects[0]!.canvas.document.stacks.raster).toHaveLength(0);
  });

  it('is disabled while editing is locked, and its command opens nothing then', async () => {
    harness.editingLocked = true;
    const { commands } = await renderActions();

    await expect.element(clearButton()).toBeDisabled();
    await act(() => commands.find((command) => command.id === 'canvas.newSession')!.handler!());
    expect(document.querySelector('[role="alertdialog"]')).toBeNull();
  });

  it('registers Clear canvas on Mod+Alt+N under its saved command id, opening the same confirm', async () => {
    const { commands, hotkeys } = await renderActions();

    expect(hotkeys.find((hotkey) => hotkey.id === 'canvas.newSession')).toMatchObject({
      defaultKeys: ['mod+alt+n'],
      title: 'Clear canvas',
    });
    await act(() => commands.find((command) => command.id === 'canvas.newSession')!.handler!());

    await expect.element(confirmDialog()).toBeVisible();
  });
});

describe('CanvasHeaderActions fit frame to masks', () => {
  it("pads the masks by the project's mask blur plus 8 before snapping outward to the grid", async () => {
    const mask: CanvasLayerContract = {
      blendMode: 'normal',
      id: 'mask',
      isEnabled: true,
      isLocked: false,
      mask: { bitmap: { height: 40, imageName: 'mask.png', width: 40 }, fill: { color: '#e07575', style: 'diagonal' } },
      name: 'Mask',
      opacity: 1,
      transform: { rotation: 0, scaleX: 1, scaleY: 1, x: 100, y: 100 },
      type: 'inpaint_mask',
    };
    const { store } = await renderActions({ canvasValues: { maskBlur: 40, snapToGrid: true }, layers: [mask] });

    await page.getByRole('button', { exact: true, name: 'Fit frame to masks' }).click();

    // Blur 40 pads 48: 52..188, on the default 8 px grid 48..192. The default blur 16 would give 72..168.
    await expect
      .poll(() => store.getState().projects[0]!.canvas.document.bbox)
      .toEqual({ height: 144, width: 144, x: 48, y: 48 });
  });
});
