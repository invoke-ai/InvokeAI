/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type { GenerateSettings, MainModelConfig } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { createExternalStoreCore } from '@platform/state/externalStoreCore';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GenerateGuidanceSection } from './GenerateGuidanceSection';

vi.mock('./GenerationUiContext', () => ({
  useGenerationUi: () => ({
    models: { status: 'loaded' },
    sectionPreferences: { sectionsOpen: { guidance: true }, setSectionOpen: vi.fn() },
  }),
}));
// The lists have their own tests; this one covers the note the section shows between them.
vi.mock('./reference-images/GenerateReferenceImagesSection', () => ({ GenerateReferenceImagesContent: () => null }));
vi.mock('./GenerateConceptsSection', async () => {
  const { isLoraCompatibleWithModel } = await import('@features/generation/core/settings');

  return {
    GenerateConceptsContent: () => null,
    isConceptInstalled: () => true,
    isCompatibleLora: (lora: { model: { base: string } }, model: { base: string }) =>
      isLoraCompatibleWithModel(lora.model, model),
  };
});

// The real catalog, so a mistyped key shows up as the key and fails the text checks below.
const i18n = createInstance();
await i18n.use(initReactI18next).init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

const krea2Model: MainModelConfig = { base: 'krea-2', format: 'diffusers', key: 'krea2', name: 'Krea-2', type: 'main' };
const editReference = {
  config: {
    image: { original: { image: { height: 512, image_name: 'pose.png', width: 512 } } },
    type: 'krea2_edit_reference_image' as const,
  },
  id: 'pose',
  isEnabled: true,
};
const poseLora = (isEnabled: boolean) => ({
  isEnabled,
  model: { base: 'krea-2', key: 'pose-lora', name: 'Pose LoRA', type: 'lora' as const },
  weight: 1,
});
const NOTE = 'Edit references need an edit LoRA (pose, depth, …); without one the model copies the reference.';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (overrides: Partial<GenerateSettings>) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const draft = createExternalStoreCore({ ...getDefaultGenerateSettings(krea2Model), ...overrides });

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <I18nextProvider i18n={i18n}>
          <GenerateGuidanceSection
            draft={draft}
            loraModels={[]}
            models={[]}
            projectId="project"
            selectedModel={krea2Model}
            onCommitImmediate={vi.fn()}
            onConceptCommit={vi.fn()}
            onReferenceCommit={vi.fn()}
          />
        </I18nextProvider>
      </ChakraProvider>
    );
  });

  return draft;
};

const note = () => [...(host?.querySelectorAll('[role="status"]') ?? [])].find((node) => node.textContent === NOTE);

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateGuidanceSection edit LoRA note', () => {
  it('announces the note while edit references run without an enabled LoRA, and drops it once one is', async () => {
    const draft = await render({ loras: [poseLora(false)], referenceImages: [editReference] });

    expect(note()).toBeDefined();

    await act(() => draft.setSnapshot({ ...draft.getSnapshot(), loras: [poseLora(true)] }));

    expect(note()).toBeUndefined();
  });

  it('stays quiet without edit references', async () => {
    await render({ loras: [], referenceImages: [{ ...editReference, isEnabled: false }] });

    expect(note()).toBeUndefined();
  });
});
