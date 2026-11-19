import type { ModelConfig } from '@features/models';

import { ChakraProvider } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { setRelationshipsSnapshotForTests } from '@features/models/data/relationshipsStore';
import { ModelsUiProvider, NO_SCHEDULER_POLICY } from '@features/models/ui/ModelsUiContext';
import { Field } from '@platform/ui/Field';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ModelSelect } from './ModelSelect';

// Browser tests run without an i18n provider, so translated copy renders as the raw key (e.g. 'models.compactRows');
// assertions below match the keys.
vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, options?: { name?: string }) => (options?.name ? `${key}:${options.name}` : key),
  }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const model = {
  base: 'sdxl',
  description: null,
  file_size: 1024,
  format: 'checkpoint',
  hash: 'hash',
  key: 'sdxl-main',
  name: 'SDXL Main',
  source: 'sdxl.safetensors',
  type: 'main',
} as ModelConfig;
const loraModel = {
  ...model,
  base: 'sd-1',
  key: 'sd1-lora',
  name: 'Detail LoRA',
  type: 'lora',
} as ModelConfig;
const secondSdxlModel = { ...model, key: 'sdxl-other', name: 'Another SDXL' } as ModelConfig;
const MODELS_UI_ADAPTER = {
  canManageModels: true,
  enableModelDescriptions: true,
  installInPlace: true,
  isProjectActive: () => true,
  managerProjectId: null,
  schedulerPolicy: NO_SCHEDULER_POLICY,
  setInstallInPlace: () => undefined,
};
const MAIN_MODEL_TYPES: ['main'] = ['main'];
const CROSS_TYPE_MODEL_TYPES: ['main', 'lora'] = ['main', 'lora'];
describe('ModelSelect loading states', () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
  });

  it('shows compatible installed models through the public picker', async () => {
    setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect modelTypes={MAIN_MODEL_TYPES} showManagerButton={false} value={null} onChange={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

    await expect.poll(() => document.querySelector('[role="option"]')?.textContent).toContain('SDXL Main');
  });

  const renderPicker = async (props: Partial<Parameters<typeof ModelSelect>[0]> = {}) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect
              modelTypes={MAIN_MODEL_TYPES}
              showManagerButton={false}
              value={null}
              onChange={vi.fn()}
              {...props}
            />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);
  };

  it('moves the active row across a group boundary and selects with Enter', async () => {
    const onChange = vi.fn();

    setModelsSnapshotForTests({ error: null, models: [model, loraModel], status: 'loaded' });
    await renderPicker({ id: 'test-keyboard', modelTypes: CROSS_TYPE_MODEL_TYPES, onChange });

    const search = document
      .querySelector<HTMLInputElement>('[role="listbox"]')!
      .closest('[data-scope="popover"]')!
      .querySelector<HTMLInputElement>('input')!;

    // sd-1 sorts before sdxl, so the LoRA leads and one step lands on the main model in the next group.
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('Detail LoRA');

    await act(() => {
      search.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'ArrowDown' }));
    });
    await expect.poll(() => document.querySelector('[data-active]')?.textContent).toContain('SDXL Main');

    await act(() => {
      search.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, cancelable: true, key: 'Enter' }));
    });

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ key: 'sdxl-main' }));
  });

  it('remembers the base filter across a close and reopen', async () => {
    const sd1Main = { ...model, base: 'sd-1', key: 'sd1-main', name: 'SD1 Main' } as ModelConfig;
    setModelsSnapshotForTests({ error: null, models: [model, sd1Main], status: 'loaded' });
    await renderPicker({ id: 'test-base-filter' });

    const chip = () =>
      Array.from(document.querySelectorAll<HTMLElement>('[data-scope="popover"] [role="button"][aria-pressed]')).find(
        (badge) => badge.textContent === 'SDXL'
      ) ?? null;
    await expect.poll(chip).not.toBeNull();
    await act(() => chip()!.click());
    await expect.poll(() => chip()?.getAttribute('aria-pressed')).toBe('true');
    await expect
      .poll(() => [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? ''))
      .not.toContainEqual(expect.stringContaining('SD1 Main'));

    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

    await expect.poll(() => chip()?.getAttribute('aria-pressed')).toBe('true');
    await expect
      .poll(() => [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? ''))
      .not.toContainEqual(expect.stringContaining('SD1 Main'));
  });

  describe('inside a Field', () => {
    const renderInField = (field: { error?: string; helpText?: string }, onChange = vi.fn()) =>
      act(() => {
        root.render(
          <ChakraProvider value={system}>
            <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
              <Field label="Base model" {...field}>
                <ModelSelect
                  id="test-field"
                  modelTypes={MAIN_MODEL_TYPES}
                  showManagerButton={false}
                  value="sdxl-main"
                  onChange={onChange}
                />
              </Field>
            </ModelsUiProvider>
          </ChakraProvider>
        );
      });
    const trigger = () => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!;
    const describedText = () =>
      (trigger().getAttribute('aria-describedby') ?? '')
        .split(' ')
        .map((id) => document.getElementById(id)?.textContent)
        .join(' ');

    it('takes its name from the field label and its description from the help text', async () => {
      setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });
      await renderInField({ helpText: 'Used for every generation' });

      await expect
        .element(page.getByRole('button', { exact: true, name: 'Base model SDXL Main SDXL' }))
        .toBeInTheDocument();
      expect(describedText()).toBe('Used for every generation');
      expect(trigger().hasAttribute('aria-invalid')).toBe(false);
    });

    it('returns focus to the trigger after a keyboard choice and after Escape', async () => {
      const onChange = vi.fn();
      setModelsSnapshotForTests({ error: null, models: [model, secondSdxlModel], status: 'loaded' });
      await renderInField({}, onChange);

      await userEvent.click(trigger());
      await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('combobox');
      await userEvent.keyboard('{ArrowUp}{Enter}');
      expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ key: 'sdxl-other' }));
      await expect.poll(() => document.activeElement).toBe(trigger());

      await userEvent.click(trigger());
      await expect.poll(() => document.activeElement?.getAttribute('role')).toBe('combobox');
      await userEvent.keyboard('{Escape}');
      await expect.poll(() => document.activeElement).toBe(trigger());
      expect(onChange).toHaveBeenCalledOnce();
    });
  });
});

describe('ModelSelect related-model ranking', () => {
  let host: HTMLDivElement;
  let root: Root;
  const vaeA = { ...model, key: 'vae-a', name: 'Alpha VAE', type: 'vae' } as ModelConfig;
  const vaeB = { ...model, key: 'vae-b', name: 'Beta VAE', type: 'vae' } as ModelConfig;
  const vaeTypes = ['vae'] as const;

  const renderVaePicker = async (props: { relatedTo?: string | null; value: string | null }) => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect modelTypes={vaeTypes} showManagerButton={false} onChange={vi.fn()} {...props} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(2);
  };
  const options = () => [...document.querySelectorAll<HTMLElement>('[role="option"]')];
  const optionNames = () => options().map((option) => option.textContent ?? '');

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    setModelsSnapshotForTests({ error: null, models: [model, vaeA, vaeB], status: 'loaded' });
    setRelationshipsSnapshotForTests({ relatedKeysByModelKey: { 'sdxl-main': ['vae-b'], 'vae-a': ['vae-b'] } });
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
    document.querySelectorAll('[data-scope="popover"]').forEach((element) => element.remove());
    setModelsSnapshotForTests({ error: null, models: [], status: 'idle' });
    setRelationshipsSnapshotForTests({ relatedKeysByModelKey: {} });
  });

  it('ranks models linked to the anchor first and names the anchor in the option', async () => {
    await renderVaePicker({ relatedTo: 'sdxl-main', value: null });

    await expect.poll(() => optionNames()[0]).toContain('Beta VAE');
    await expect.element(options()[0]!).toHaveAccessibleName(/^Beta VAE\s*, models\.linkedToModel:SDXL Main/);
    await expect.element(options()[1]!).toHaveAccessibleName(/^Alpha VAE/);
    await expect.element(options()[1]!).not.toHaveAccessibleName(/linked/i);
  });
});
