import type { ControlLayerRejection } from '@workbench/widgets/canvas/invoke/prepareCanvasInvocation';

import { FocusRegionProvider } from '@workbench/focusRegions';
import { createTestFocusController } from '@workbench/focusRegions.testing';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';

import { useRegisterFirstPartyCommands } from './firstPartyCommands';

const handlers = vi.hoisted(() => new Map<string, () => unknown>());
const submitActiveInvocation = vi.hoisted(() =>
  vi.fn<(args: { formatControlLayerError: (rejection: ControlLayerRejection) => string }) => Promise<void>>(() =>
    Promise.resolve()
  )
);

vi.mock('@workbench/activeInvocationSubmission', () => ({ submitActiveInvocation }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useWorkbenchCommands: () => ({ layout: {}, notifications: {}, queue: {}, widgets: {} }),
  useWorkbenchExtensions: () => ({
    commands: {
      register: ({ handler, id }: { handler: () => unknown; id: string }) => {
        handlers.set(id, handler);
        return () => handlers.delete(id);
      },
    },
  }),
  useWorkbenchQueries: () => ({ getSnapshot: () => ({}) }),
}));
vi.mock('@tanstack/react-query', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useQueryClient: () => ({}),
}));
vi.mock('@features/models', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  ensureModelsLoaded: () => Promise.resolve(),
}));

const english = createInstance();
const focus = createTestFocusController();
await english.use(initReactI18next).init({
  initAsync: false,
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
const Commands = () => {
  useRegisterFirstPartyCommands();
  return null;
};

afterEach(async () => {
  await act(() => root?.unmount());
  root = null;
  handlers.clear();
  submitActiveInvocation.mockClear();
});

it('words a blocked control layer from the locale when Invoke runs from its hotkey', async () => {
  const host = document.createElement('div');
  root = createRoot(host);
  await act(() =>
    root?.render(
      <I18nextProvider i18n={english}>
        <FocusRegionProvider controller={focus}>
          <Commands />
        </FocusRegionProvider>
      </I18nextProvider>
    )
  );

  await act(async () => {
    await handlers.get('app.invoke')?.();
  });

  const [args] = submitActiveInvocation.mock.calls[0] ?? [];
  expect(
    args?.formatControlLayerError({ code: 'switch_adapter_kind', layerName: 'Sketch', suggestedKind: 'anima_lllite' })
  ).toBe(
    'Control layer "Sketch": Switch the adapter type to Anima ControlNet-LLLite to use this layer with the selected model.'
  );
});
