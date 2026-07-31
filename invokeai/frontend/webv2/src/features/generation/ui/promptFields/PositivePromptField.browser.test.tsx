/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { PositivePromptField } from '@features/generation/ui/promptFields/PositivePromptField';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, expect, it, vi } from 'vitest';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    gallery: { selectedImage: null },
    models: { catalog: [], ensureLoaded: vi.fn() },
    notifications: { reportError: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [] },
  }),
}));

vi.mock('@features/generation/data/wildcards', () => ({
  createWildcard: vi.fn(),
  deleteWildcard: vi.fn(),
  invalidateWildcardDependents: vi.fn(),
  updateWildcard: vi.fn(),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

it('keeps positive prompt actions enabled when stale view mode has no active template', async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <QueryClientProvider client={new QueryClient()}>
          <ChakraProvider value={system}>
            {/* The prompt box is a gallery-image drop target, so it lives inside
                the shell's DndContext wherever it is really rendered. */}
            <DndContext>
              <PositivePromptField
                heightPx={96}
                isTemplateViewMode
                loras={[]}
                projectId="project-1"
                selectedModel={undefined}
                showSyntaxHighlighting={false}
                value="a cat"
                onChange={vi.fn()}
                onResizeEnd={vi.fn()}
                onUsePrompt={vi.fn()}
              />
            </DndContext>
          </ChakraProvider>
        </QueryClientProvider>
      </I18nextProvider>
    );
  });

  expect(host.querySelector('textarea')?.readOnly).toBe(false);
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.addPromptTrigger"]')?.disabled).toBe(
    false
  );
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.expandPrompt"]')?.disabled).toBe(false);
  expect(host.querySelector<HTMLButtonElement>('[aria-label="widgets.generate.imageToPrompt"]')?.disabled).toBe(false);
});
