/* oxlint-disable react-perf/jsx-no-new-object-as-prop, react-perf/jsx-no-new-array-as-prop */
import type {
  GenerateLora,
  GenerateModelConfig,
  GenerateSettings,
  LoraModelConfig,
} from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import { seedArchitectureCapabilities } from '@features/generation/core/architectureCapabilities.testing';
import { getDefaultGenerateSettings } from '@features/generation/core/baseGenerationPolicies';
import { flushGenerateDrafts } from '@features/generation/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { GenerateSettingsUpdate } from './generateDebounce';

import { GenerateConceptsContent } from './GenerateConceptsSection';
import { GenerationUiProvider, type GenerationUiAdapter } from './GenerationUiContext';
import { GenerateIssuesProvider } from './shared/GenerateIssues';

vi.mock('react-i18next', () => {
  const t = (key: string) => key;
  return { useTranslation: () => ({ i18n: { resolvedLanguage: 'en' }, t }) };
});

seedArchitectureCapabilities();

const LORA_MODEL: LoraModelConfig = { base: 'sdxl', key: 'lora-1', name: 'Ink Wash', type: 'lora' };
const ADDED_LORA_MODEL: LoraModelConfig = { base: 'sdxl', key: 'lora-2', name: 'Chalk', type: 'lora' };
const MAIN_MODEL: GenerateModelConfig = { base: 'sdxl', key: 'main-1', name: 'Base', type: 'main' };
const SD3_MODEL: GenerateModelConfig = { base: 'sd-3', key: 'sd3', name: 'SD3', type: 'main' };
const LORA_MODELS = [LORA_MODEL];
const LORA: GenerateLora = { isEnabled: true, model: LORA_MODEL, weight: 0.75 };
const SETTINGS: GenerateSettings = { ...getDefaultGenerateSettings(), loras: [LORA] };
const ADAPTER = {
  models: {
    ModelSelect: ({ onChange }: { onChange: (model: LoraModelConfig) => void }) => (
      <button data-testid="concept-picker" type="button" onClick={() => onChange(ADDED_LORA_MODEL)}>
        Add concept
      </button>
    ),
    getBaseColorPalette: () => 'gray',
    getBaseLabel: (base: string) => base,
    getImageUrl: () => '',
  },
} as unknown as GenerationUiAdapter;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const render = async (selectedModel: GenerateModelConfig = MAIN_MODEL, settings = SETTINGS) => {
  const onCommit = vi.fn<(update: GenerateSettingsUpdate) => void>();

  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const renderProject = (projectId: string) =>
    act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={ADAPTER}>
            <GenerateConceptsContent
              loraModels={LORA_MODELS}
              loras={settings.loras}
              projectId={projectId}
              selectedModel={selectedModel}
              onCommit={onCommit}
            />
          </GenerationUiProvider>
        </ChakraProvider>
      )
    );
  await renderProject('project-1');

  return { onCommit, renderProject, row: host };
};

/** The settings a recorded commit produces from the rendered ones. */
const applied = (update: GenerateSettingsUpdate | undefined): GenerateLora | undefined =>
  (typeof update === 'function' ? update(SETTINGS) : { ...SETTINGS, ...update }).loras[0];

const stepWeight = (row: HTMLElement) =>
  act(() => {
    row
      .querySelector('[role="slider"]')
      ?.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowRight' }));
  });

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateConceptsContent', () => {
  it('discards a concept menu when switching projects with the same concept', async () => {
    const { onCommit, renderProject, row } = await render();

    await stepWeight(row);
    expect(row.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.8');

    await act(() => {
      row
        .querySelector('[data-list-primary]')
        ?.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
    });
    await expect.poll(() => document.querySelector('[role="menu"][data-state="open"]')).not.toBeNull();

    await renderProject('project-2');

    expect(document.querySelector('[role="menu"][data-state="open"]')).toBeNull();
    await renderProject('project-1');
    expect(document.querySelector('[role="menu"][data-state="open"]')).toBeNull();
    act(() => flushGenerateDrafts());
    expect(onCommit).not.toHaveBeenCalled();
  });

  it('adds a concept to the latest settings rather than the rendered list', async () => {
    const { onCommit, row } = await render();

    await act(() => row.querySelector<HTMLElement>('[data-testid="concept-picker"]')?.click());

    expect(onCommit).toHaveBeenCalledTimes(1);
    const update = onCommit.mock.calls[0]?.[0];
    expect(typeof update).toBe('function');
    // A weight that landed after this render must survive the add.
    const latest = { ...SETTINGS, loras: [{ ...LORA, weight: 0.9 }] };
    expect((update as (settings: GenerateSettings) => GenerateSettings)(latest).loras).toEqual([
      { ...LORA, weight: 0.9 },
      { isEnabled: true, model: ADDED_LORA_MODEL, weight: expect.any(Number) },
    ]);
  });

  it('ignores a picked concept the selected model cannot load', async () => {
    const fluxModel: GenerateModelConfig = { base: 'flux', key: 'flux', name: 'FLUX', type: 'main' };
    const { onCommit, row } = await render(fluxModel);

    await act(() => row.querySelector<HTMLElement>('[data-testid="concept-picker"]')?.click());

    const update = onCommit.mock.calls[0]?.[0] as (settings: GenerateSettings) => GenerateSettings;
    expect(update(SETTINGS)).toBe(SETTINGS);
  });

  it('holds weight steps as a draft and commits them once the debounce settles', async () => {
    const { onCommit, row } = await render();

    await stepWeight(row);
    await stepWeight(row);

    expect(row.querySelector('[role="slider"]')?.getAttribute('aria-valuenow')).toBe('0.85');
    expect(onCommit).not.toHaveBeenCalled();

    await expect.poll(() => onCommit.mock.calls.length, { timeout: 1000 }).toBe(1);
    expect(applied(onCommit.mock.calls[0]?.[0])?.weight).toBe(0.85);
  });

  it('commits a pending weight when drafts are flushed', async () => {
    const { onCommit, row } = await render();

    await stepWeight(row);
    act(() => flushGenerateDrafts());

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(applied(onCommit.mock.calls[0]?.[0])?.weight).toBe(0.8);
  });

  it('commits toggles immediately rather than through the weight draft', async () => {
    const { onCommit, row } = await render();

    await act(() => row.querySelector<HTMLElement>('[data-scope="switch"][data-part="control"]')?.click());

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(applied(onCommit.mock.calls[0]?.[0])).toMatchObject({ isEnabled: false, weight: 0.75 });
  });

  it('offers the concept picker for a model family that loads LoRAs', async () => {
    const { row } = await render(MAIN_MODEL, { ...SETTINGS, loras: [] });

    expect(row.querySelector('[data-testid="concept-picker"]')).not.toBeNull();
    expect(row.textContent).toContain('widgets.generate.addConceptsHelp');
    expect(row.textContent).not.toContain('widgets.generate.conceptsUnsupported');
  });

  it('explains instead of offering an always-empty picker when the model cannot use concepts', async () => {
    const { row } = await render(SD3_MODEL, { ...SETTINGS, loras: [] });

    expect(row.querySelector('[data-testid="concept-picker"]')).toBeNull();
    expect(row.textContent).toContain('widgets.generate.conceptsUnsupported');
    expect(row.textContent).not.toContain('widgets.generate.addConceptsHelp');
  });

  it('keeps concepts from another model removable when the model cannot use concepts', async () => {
    const { row, onCommit } = await render(SD3_MODEL);

    expect(row.textContent).toContain('Ink Wash');
    expect(row.textContent).toContain('widgets.generate.incompatible');

    const remove = row.querySelector<HTMLButtonElement>('button[aria-label="widgets.generate.removeConceptNamed"]');

    expect(remove).not.toBeNull();
    await act(() => {
      remove?.click();
    });

    expect(onCommit).toHaveBeenCalledTimes(1);
    expect(applied(onCommit.mock.calls[0]?.[0])).toBeUndefined();
  });

  it('marks a same-family concept incompatible when the model cannot use concepts', async () => {
    const sd3Lora: LoraModelConfig = { base: 'sd-3', key: 'sd3-lora', name: 'Glow', type: 'lora' };
    const { row } = await render(SD3_MODEL, {
      ...SETTINGS,
      loras: [{ isEnabled: true, model: sd3Lora, weight: 0.8 }],
    });

    expect(row.textContent).toContain('widgets.generate.incompatible');
  });
});

describe('a concept that is not installed', () => {
  it('reads as blocked: its saved state, a locked switch, an error badge, and the issue with its Remove fix', async () => {
    const missing: LoraModelConfig = { base: 'sdxl', key: 'gone', name: 'Gone', type: 'lora' };
    const applyFix = vi.fn();
    const issue = {
      fix: { kind: 'removeConcept' as const, modelKey: missing.key },
      id: `concepts:notInstalled:${missing.key}`,
      message: 'Concept "Gone" is not installed. Remove it, or reinstall it.',
      messageKey: 'widgets.generate.issues.conceptNotInstalled',
      target: { field: 'concepts' as const, section: 'guidance' as const },
    };
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await act(() =>
      root?.render(
        <ChakraProvider value={system}>
          <GenerationUiProvider
            adapter={{ models: { ...ADAPTER.models, status: 'loaded' } } as unknown as GenerationUiAdapter}
          >
            <GenerateIssuesProvider applyFix={applyFix} issues={[issue]}>
              <GenerateConceptsContent
                loraModels={LORA_MODELS}
                loras={[LORA, { isEnabled: true, model: missing, weight: 0.5 }]}
                projectId="project-1"
                selectedModel={MAIN_MODEL}
                onCommit={vi.fn()}
              />
            </GenerateIssuesProvider>
          </GenerationUiProvider>
        </ChakraProvider>
      )
    );
    const missingRow = host.querySelector('[role="group"][aria-label="Gone"]')!.closest('[role="listitem"]')!;
    const installedRow = host.querySelector('[role="group"][aria-label="Ink Wash"]')!.closest('[role="listitem"]')!;
    const toggle = missingRow.querySelector<HTMLElement>('[role="switch"], input[type="checkbox"]')!;

    expect(missingRow.textContent).toContain('widgets.generate.notInstalled');
    expect(toggle.matches(':checked') || toggle.getAttribute('aria-checked') === 'true').toBe(true);
    expect(toggle.matches(':disabled') || toggle.getAttribute('aria-disabled') === 'true').toBe(true);
    expect(missingRow.textContent).toContain('widgets.generate.issues.conceptNotInstalled');
    expect(installedRow.querySelector('[data-issue-fix]')).toBeNull();

    await act(() => missingRow.querySelector<HTMLButtonElement>('[data-issue-fix]')!.click());

    expect(applyFix).toHaveBeenCalledExactlyOnceWith(issue.fix);
  });
});
