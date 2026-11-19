import type { CanvasVectorLayerContract } from '@workbench/canvas-engine/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createVectorLayer } from '@workbench/canvas-engine/api';
import { documentFrom } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createTestStubRasterBackend } from '@workbench/canvas-engine/render/raster.testStub';
import { createCanvasEngine } from '@workbench/canvas-operations/createCanvasEngine';
import { createCanvasProjectMutationPort } from '@workbench/canvasProjectMutationPort';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createWorkbenchStore } from '@workbench/workbenchStore';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeAll, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { VectorLayerSettings } from './VectorLayerSettings';

const i18n = createInstance();
let root: Root | null = null;
let host: HTMLDivElement | null = null;
let engine: ReturnType<typeof createCanvasEngine> | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(async () => {
  const translation = await fetch('/locales/en.json').then((response) => response.json());
  await i18n.use(initReactI18next).init({ initAsync: false, lng: 'en', resources: { en: { translation } } });
});

afterEach(async () => {
  await act(() => root?.unmount());
  engine?.lifecycle.dispose();
  host?.remove();
  root = null;
  host = null;
  engine = null;
});

it('edits and undoes layer appearance independently of another vector layer and the brush', async () => {
  const a = {
    ...createVectorLayer({ id: 'a', name: 'A' }),
    strokeColor: '#ff0000',
    strokeWidth: 17,
    fillColor: '#abcdef',
    fillEnabled: true,
  };
  const b = {
    ...createVectorLayer({ id: 'b', name: 'B' }),
    strokeColor: '#0000ff',
    strokeWidth: 3,
    fillColor: '#00ff00',
    fillEnabled: true,
  };
  const initial = createInitialWorkbenchState();
  const project = initial.projects[0]!;
  const store = createWorkbenchStore({
    ...initial,
    projects: [{ ...project, canvas: { ...project.canvas, document: documentFrom([a, b], 'a') } }],
  });
  engine = createCanvasEngine({
    backend: createTestStubRasterBackend(),
    imageResolver: () => Promise.resolve(new Blob()),
    ensureProjectOnServer: () => Promise.resolve(),
    reportError: () => undefined,
    mutationPort: createCanvasProjectMutationPort(store, project.id),
    projectId: project.id,
  });
  const liveEngine = engine;
  const document = () => store.queries.getSnapshot().activeProject.canvas.document;
  const layer = (id: string) => getDocumentLayer(document(), id) as CanvasVectorLayerContract;
  const Harness = () => {
    const doc = useSyncExternalStore(store.subscribe, document);
    const selected = getDocumentLayer(doc, doc.selectedLayerId!) as CanvasVectorLayerContract;
    return <VectorLayerSettings key={selected.id} engine={liveEngine} layer={selected} />;
  };
  host = globalThis.document.createElement('div');
  host.style.width = '340px';
  globalThis.document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness />
        </ChakraProvider>
      </I18nextProvider>
    )
  );
  const brush = liveEngine.interaction.get('brushOptions');
  const width = page.getByRole('slider', { name: 'Stroke width', exact: true });
  await expect.element(width).toHaveAttribute('aria-valuenow', '17');
  await width.element().focus();
  await act(() => userEvent.keyboard('{ArrowRight}'));
  expect(layer('a').strokeWidth).toBe(18);
  expect(layer('b')).toEqual(b);
  expect(liveEngine.interaction.get('brushOptions')).toEqual(brush);
  await act(() => liveEngine.history.undo());
  expect(layer('a').strokeWidth).toBe(17);
  await act(() => liveEngine.history.redo());
  expect(layer('a').strokeWidth).toBe(18);

  await act(() => page.getByRole('button', { name: 'Trace color', exact: true }).click());
  const area = globalThis.document.querySelector<HTMLElement>('[data-scope="color-picker"][data-part="area"]')!;
  const bounds = area.getBoundingClientRect();
  const at = {
    clientX: bounds.left + bounds.width * 0.4,
    clientY: bounds.top + bounds.height * 0.4,
    pointerId: 1,
    button: 0,
    isPrimary: true,
    bubbles: true,
  };
  await act(() => area.dispatchEvent(new PointerEvent('pointerdown', at)));
  await vi.waitFor(() => expect(layer('a').strokeColor).not.toBe('#ff0000'));
  const previewed = layer('a').strokeColor;
  await act(() => globalThis.document.dispatchEvent(new PointerEvent('pointerup', at)));
  await act(() => userEvent.keyboard('{Escape}'));
  expect(layer('a').strokeColor).toBe(previewed);
  expect(layer('b')).toEqual(b);
  expect(liveEngine.interaction.get('brushOptions')).toEqual(brush);
  await act(() => liveEngine.history.undo());
  expect(layer('a').strokeColor).toBe('#ff0000');
  await act(() => liveEngine.history.redo());
  expect(layer('a').strokeColor).toBe(previewed);

  await act(() => page.getByRole('button', { name: 'Fill', exact: true }).click());
  expect(layer('a').fillEnabled).toBe(false);
  expect(layer('b').fillEnabled).toBe(true);
  await act(() => liveEngine.history.undo());
  expect(layer('a').fillEnabled).toBe(true);

  await act(() => store.commands.canvas.apply(project.id, { type: 'setCanvasSelectedLayer', id: 'b' }));
  await expect.element(width).toHaveAttribute('aria-valuenow', '3');
  await act(() => liveEngine.interaction.set('brushOptions', { ...brush, color: '#ff00ff', size: 80 }));
  expect(layer('a').strokeColor).toBe(previewed);
  expect(layer('a').strokeWidth).toBe(18);
  expect(layer('b')).toEqual(b);
  await act(() => store.commands.canvas.apply(project.id, { type: 'setCanvasSelectedLayer', id: 'a' }));
  await expect.element(width).toHaveAttribute('aria-valuenow', '18');
});
