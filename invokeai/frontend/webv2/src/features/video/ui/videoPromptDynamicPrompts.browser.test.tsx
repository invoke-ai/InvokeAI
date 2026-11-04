import type { DynamicPromptsConfig } from '@features/generation/settings';

import { ChakraProvider } from '@chakra-ui/react';
import { DndContext } from '@dnd-kit/core';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { VideoPromptFields } from './VideoFormFields';
import { VideoUiProvider, type VideoUiAdapter } from './VideoUiContext';

const i18n = i18next.createInstance();

await i18n.use(initReactI18next).init({ fallbackLng: 'en', lng: 'en', resources: { en: { translation: {} } } });

const parseDynamicPrompts = vi.hoisted(() => vi.fn());

vi.mock('@features/generation/ui/GenerationUiContext', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useGenerationUi: () => ({
    capabilities: { canManagePromptTemplates: false, canManageSharedSystemPrompts: false },
    gallery: { selectedImage: null },
    models: { catalog: [], ensureLoaded: vi.fn() },
    notifications: { reportError: vi.fn(), undoable: vi.fn() },
    project: { activeProjectId: 'project-1' },
    promptHistory: { clear: vi.fn(), items: [], remove: vi.fn(), restore: vi.fn() },
  }),
}));
vi.mock('@features/generation/data/promptUtilities', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  parseDynamicPrompts,
}));
vi.mock('@features/generation/data/wildcards', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  wildcardsQueryOptions: () => ({ queryFn: () => Promise.resolve([]), queryKey: ['generation', 'wildcards'] }),
}));

const adapter = {
  getUploadBoardId: () => Promise.resolve('none'),
  notifications: { info: () => undefined, undoable: () => undefined },
  patchValues: vi.fn(),
  readValues: () => ({}),
} as unknown as VideoUiAdapter;

const NO_LORAS: [] = [];
const DYNAMIC_PROMPTS: DynamicPromptsConfig = {
  combinatorial: true,
  enabled: true,
  maxPrompts: 10,
  resample: true,
  sampleSeed: 0,
  seedBehaviour: 'per-iteration',
};

let host: HTMLDivElement;
let root: Root;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  parseDynamicPrompts.mockReset();
  parseDynamicPrompts.mockResolvedValue({ error: null, prompts: ['wall painted'] });
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('Video prompt dynamic prompts', () => {
  it('stores turning dynamic prompts off as the Video setting', async () => {
    const onPatchValues = vi.fn();

    await act(() =>
      root.render(
        <I18nextProvider i18n={i18n}>
          <QueryClientProvider client={new QueryClient()}>
            <ChakraProvider value={system}>
              <DndContext>
                <VideoUiProvider adapter={adapter}>
                  <VideoPromptFields
                    batchCount={1}
                    dynamicPrompts={DYNAMIC_PROMPTS}
                    expandPromptSuggestion={null}
                    isSeedFixed={false}
                    loras={NO_LORAS}
                    model={null}
                    negativePrompt=""
                    negativePromptEnabled
                    negativePromptHeightPx={56}
                    negativeVisible={false}
                    positivePrompt="wall painted #ff0000"
                    positivePromptHeightPx={96}
                    projectId="project-1"
                    showSyntaxHighlighting={false}
                    onPatchValues={onPatchValues}
                  />
                </VideoUiProvider>
              </DndContext>
            </ChakraProvider>
          </QueryClientProvider>
        </I18nextProvider>
      )
    );

    // The notice under the prompt offers the switch, as the panel's does; both write through the same handler.
    const turnOff = [...document.querySelectorAll<HTMLButtonElement>('button')].find(
      (button) => button.textContent === 'widgets.generate.dynamicPrompts.turnOff'
    );

    expect(turnOff).toBeDefined();
    await act(() => turnOff!.click());

    expect(onPatchValues).toHaveBeenCalledWith({ dynamicPromptsEnabled: false });
  });
});
