import type { ModelConfig } from '@features/models';

import { ChakraProvider, Field as ChakraField } from '@chakra-ui/react';
import { setModelsSnapshotForTests } from '@features/models/data/modelsStore';
import { ModelsUiProvider } from '@features/models/ui/ModelsUiContext';
import { auditAccessibility, contrastOffenderTexts } from '@platform/browser/auditAccessibility.testing';
import { Field } from '@platform/ui/Field';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { ModelSelect } from './ModelSelect';

// Browser tests run without an i18n provider, so translated copy renders as
// the raw key (e.g. 'models.compactRows'); assertions below match the keys.
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

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
  isProjectActive: () => true,
  managerProjectId: null,
};
const MAIN_MODEL_TYPES: ['main'] = ['main'];
const CROSS_TYPE_MODEL_TYPES: ['main', 'lora'] = ['main', 'lora'];

const CONTROLNET_TYPES = ['controlnet'] as const;

const LORA_TYPES = ['lora'] as const;
const ADDED_LORA_KEYS: ReadonlySet<string> = new Set(['sd1-lora']);

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

  it('says every compatible model is already added rather than that none are installed', async () => {
    setModelsSnapshotForTests({ error: null, models: [loraModel], status: 'loaded' });

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect
              excludeKeys={ADDED_LORA_KEYS}
              modelTypes={LORA_TYPES}
              scopeLabel="concepts"
              showManagerButton={false}
              value={null}
              onChange={vi.fn()}
            />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    const trigger = host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!;
    expect(trigger.disabled).toBe(true);
    expect(trigger.textContent).toContain('models.scopeAllAdded');
  });

  it('disables the trigger instead of opening an empty list when nothing compatible is installed', async () => {
    setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect modelTypes={CONTROLNET_TYPES} showManagerButton={false} value={null} onChange={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });

    const trigger = host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!;
    expect(trigger.disabled).toBe(true);
    expect(trigger.textContent).toContain('scopeNoCompatibleInstalled');
    await act(() => trigger.click());
    // Closed: the picker computes no options, and the popover never opens.
    expect(document.querySelectorAll('[role="option"]')).toHaveLength(0);
    expect(document.querySelector('[data-scope="popover"][data-part="content"][data-state="open"]')).toBeNull();

    // While the library is still loading there is nothing to conclude yet.
    setModelsSnapshotForTests({ error: null, models: [], status: 'loading' });
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
            <ModelSelect modelTypes={CONTROLNET_TYPES} showManagerButton={false} value={null} onChange={vi.fn()} />
          </ModelsUiProvider>
        </ChakraProvider>
      );
    });
    expect(host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')!.disabled).toBe(false);
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

  const optionTexts = (): string[] =>
    [...document.querySelectorAll('[role="option"]')].map((option) => option.textContent ?? '');

  // Cross-type pickers must not repeat Main Models headings for every base.
  it('groups a cross-type picker by base alone, with no type header', async () => {
    setModelsSnapshotForTests({ error: null, models: [model, loraModel], status: 'loaded' });
    await renderPicker({ id: 'test-cross-type', modelTypes: CROSS_TYPE_MODEL_TYPES });

    const listbox = document.querySelector('[role="listbox"]')!;

    expect(listbox.textContent).toContain('Stable Diffusion 1.x');
    expect(listbox.textContent).toContain('Stable Diffusion XL');
    expect(listbox.textContent).not.toContain('Main Models');
    // Type is on the row instead, so the two rows stay distinguishable.
    expect(optionTexts().join(' ')).toContain('LoRA');
  });

  it('moves the active row across a group boundary and selects with Enter', async () => {
    const onChange = vi.fn();

    setModelsSnapshotForTests({ error: null, models: [model, loraModel], status: 'loaded' });
    await renderPicker({ id: 'test-keyboard', modelTypes: CROSS_TYPE_MODEL_TYPES, onChange });

    const search = document
      .querySelector<HTMLInputElement>('[role="listbox"]')!
      .closest('[data-scope="popover"]')!
      .querySelector<HTMLInputElement>('input')!;

    // sd-1 sorts before sdxl, so the LoRA leads and one step lands on the main
    // model in the next group.
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

  it('remembers the compact row density across a close and reopen', async () => {
    setModelsSnapshotForTests({ error: null, models: [model, secondSdxlModel], status: 'loaded' });
    await renderPicker({ id: 'test-compact' });

    const toggle = document.querySelector<HTMLButtonElement>('button[aria-label="models.compactRows"]')!;

    expect(toggle).not.toBeNull();
    await act(() => toggle.click());
    await expect.poll(() => document.querySelector('button[aria-label="models.fullRows"]')).not.toBeNull();

    // Close and reopen: the preference is stored per picker id, not per mount.
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());

    await expect.poll(() => document.querySelector('button[aria-label="models.fullRows"]')).not.toBeNull();
  });

  it('keeps its scroll and rows while a parent re-renders with inline filter and types', async () => {
    const many = Array.from(
      { length: 60 },
      (_, index) => ({ ...model, key: `sdxl-${index}`, name: `SDXL ${String(index).padStart(2, '0')}` }) as ModelConfig
    );
    setModelsSnapshotForTests({ error: null, models: many, status: 'loaded' });
    // Like Generate's callers: a fresh filter and type list on every parent render.
    const renderParent = () =>
      act(() => {
        root.render(
          <ChakraProvider value={system}>
            <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
              <ModelSelect
                // oxlint-disable-next-line react-perf/jsx-no-new-function-as-prop -- the identity churn under test
                filter={(candidate) => candidate.base === 'sdxl'}
                id="test-inline"
                // oxlint-disable-next-line react-perf/jsx-no-new-array-as-prop -- the identity churn under test
                modelTypes={['main']}
                showManagerButton={false}
                value={null}
                onChange={vi.fn()}
              />
            </ModelsUiProvider>
          </ChakraProvider>
        );
      });
    await renderParent();
    await act(() => host.querySelector<HTMLButtonElement>('[aria-haspopup="listbox"]')?.click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBeGreaterThan(0);

    const viewport = document
      .querySelector<HTMLElement>('[role="listbox"]')!
      .closest<HTMLElement>('[data-part="viewport"]')!;
    const scrollTo = (offset: number) => () => {
      viewport.scrollTop = offset;
      viewport.dispatchEvent(new Event('scroll'));

      return viewport.scrollTop === offset;
    };
    await expect.poll(() => act(scrollTo(400))).toBe(true);
    // A row of the scrolled-to window, not the active first option, which stays mounted wherever the list scrolls.
    const findWindowRow = () =>
      [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((option) =>
        option.textContent?.startsWith('SDXL 15')
      ) ?? null;
    await expect.poll(findWindowRow).not.toBeNull();
    const row = findWindowRow()!;

    await renderParent();
    await renderParent();

    expect(viewport.scrollTop).toBe(400);
    expect(document.getElementById(row.id)).toBe(row);
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

      // The label names the control; the selected model and its base follow as its value.
      await expect
        .element(page.getByRole('button', { exact: true, name: 'Base model SDXL Main SDXL' }))
        .toBeInTheDocument();
      expect(describedText()).toBe('Used for every generation');
      expect(trigger().hasAttribute('aria-invalid')).toBe(false);
    });

    it('reports the field error as invalid and describes the trigger with it', async () => {
      setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });
      await renderInField({ error: 'Choose an installed model' });

      await expect.poll(describedText).toBe('Choose an installed model');
      expect(trigger().getAttribute('aria-invalid')).toBe('true');
    });

    it('names itself from content inside a field host that renders no label', async () => {
      setModelsSnapshotForTests({ error: null, models: [model], status: 'loaded' });
      await act(() => {
        root.render(
          <ChakraProvider value={system}>
            <ModelsUiProvider adapter={MODELS_UI_ADAPTER}>
              <ChakraField.Root>
                <ModelSelect
                  modelTypes={MAIN_MODEL_TYPES}
                  showManagerButton={false}
                  value="sdxl-main"
                  onChange={vi.fn()}
                />
              </ChakraField.Root>
            </ModelsUiProvider>
          </ChakraProvider>
        );
      });

      expect(trigger().hasAttribute('aria-labelledby')).toBe(false);
      await expect.element(page.getByRole('button', { exact: true, name: 'SDXL Main SDXL' })).toBeInTheDocument();
    });

    it('mounts the picker only while open, with its active option named in the first commit', async () => {
      setModelsSnapshotForTests({ error: null, models: [model, secondSdxlModel], status: 'loaded' });
      await renderInField({});
      expect(document.querySelector('input[role="combobox"]')).toBeNull();

      // Checked synchronously after the opening commit, before any resize observation can re-render the list.
      await act(() => trigger().click());
      const search = document.querySelector<HTMLInputElement>('input[role="combobox"]')!;
      expect(document.getElementById(search.getAttribute('aria-activedescendant')!)?.textContent).toContain(
        'SDXL Main'
      );

      await act(() => trigger().click());
      await expect.poll(() => document.querySelector('input[role="combobox"]')).toBeNull();
    });

    it('keeps the popup search out of the field and passes an audit while open', async () => {
      setModelsSnapshotForTests({ error: null, models: [model, secondSdxlModel], status: 'loaded' });
      await renderInField({ error: 'Choose an installed model' });
      await act(() => trigger().click());

      const search = await vi.waitFor(() => document.querySelector<HTMLInputElement>('input[role="combobox"]')!);
      // The field's label and error belong to the trigger, not to the search box inside its popup.
      expect(search.id).not.toBe(document.querySelector('label')!.htmlFor);
      expect(search.hasAttribute('aria-invalid')).toBe(false);
      expect(search.hasAttribute('aria-describedby')).toBe(false);
      expect(document.getElementById(search.getAttribute('aria-controls')!)?.getAttribute('role')).toBe('listbox');
      expect(document.getElementById(search.getAttribute('aria-activedescendant')!)?.textContent).toContain(
        'SDXL Main'
      );

      const popup = document.querySelector('[data-scope="popover"][data-part="content"]')!;
      const violations = [...(await auditAccessibility(host)), ...(await auditAccessibility(popup))];
      expect(violations.filter((violation) => violation.id !== 'color-contrast')).toEqual([]);
      // Known theme-token offenders, pinned so that any new failing node (or a fixed one) changes the list: the
      // subtle file-size captions and the invalid trigger's red value text (its middle-truncated tail).
      expect(contrastOffenderTexts(violations)).toEqual(['1.0 kB', '1.0 kB', 'DXL Main']);
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
