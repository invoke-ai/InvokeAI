/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { Combobox, type ComboboxOption } from './Combobox';
import { Field } from './Field';

const options: ComboboxOption[] = [
  { label: 'Euler Ancestral', value: 'euler_a' },
  { label: 'DPM++ 2M', value: 'dpmpp_2m' },
  { label: 'UniPC', searchText: 'popular sampler', value: 'unipc' },
];

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: {
          noOptionsFound: 'No matches',
          openSelector: 'Open selector',
          searchOptions: 'Search…',
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const interact = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

const Harness = ({
  disabled = false,
  filterOptions = false,
  onInputValueChange,
  onChange,
}: {
  disabled?: boolean;
  filterOptions?: boolean;
  onInputValueChange?: (value: string) => void;
  onChange: (value: string) => void;
}) => {
  const [value, setValue] = useState('euler_a');
  const [search, setSearch] = useState('');
  const visibleOptions = filterOptions
    ? options.filter((option) => option.label.toLocaleLowerCase().includes(search.toLocaleLowerCase()))
    : options;

  return (
    <Combobox
      aria-label="Scheduler"
      disabled={disabled}
      options={visibleOptions}
      value={value}
      onInputValueChange={
        filterOptions
          ? (nextSearch) => {
              setSearch(nextSearch);
              onInputValueChange?.(nextSearch);
            }
          : undefined
      }
      onValueChange={(nextValue) => {
        setValue(nextValue);
        onChange(nextValue);
      }}
    />
  );
};

const renderCombobox = async (disabled = false, filterOptions = false) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onChange = vi.fn();
  const onInputValueChange = vi.fn();

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness
            disabled={disabled}
            filterOptions={filterOptions}
            onChange={onChange}
            onInputValueChange={onInputValueChange}
          />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return { input: host.querySelector<HTMLInputElement>('input[role="combobox"]')!, onChange, onInputValueChange };
};

const setInputValue = async (input: HTMLInputElement, value: string) => {
  const valueSetter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

  await interact(() => {
    valueSetter?.call(input, value);
    input.dispatchEvent(new InputEvent('input', { bubbles: true, data: value, inputType: 'insertText' }));
  });
};

afterEach(async () => {
  await interact(() => root?.unmount());
  document.querySelectorAll('[data-scope="combobox"][data-part="positioner"]').forEach((element) => element.remove());
  host?.remove();
  host = null;
  root = null;
});

describe('Combobox', () => {
  it('filters labels and values case-insensitively and reports empty results', async () => {
    const { input } = await renderCombobox();

    expect(input.value).toBe('Euler Ancestral');
    await interact(() => input.click());
    await setInputValue(input, 'DPM');
    expect(document.querySelectorAll('[role="option"]')).toHaveLength(1);
    expect(document.querySelector('[role="option"]')?.textContent).toContain('DPM++ 2M');

    await setInputValue(input, 'not-a-scheduler');
    expect(document.body.textContent).toContain('No matches');
  });

  it('keeps an option that matches secondary server-search text', async () => {
    const { input } = await renderCombobox();

    await interact(() => input.click());
    await setInputValue(input, 'popular');

    expect(document.querySelectorAll('[role="option"]')).toHaveLength(1);
    expect(document.querySelector('[role="option"]')?.textContent).toContain('UniPC');
  });

  it('supports controlled mouse and keyboard selection with selected-state indication', async () => {
    const { input, onChange } = await renderCombobox();

    await interact(() => input.click());
    expect(document.querySelector('[role="option"][data-state="checked"]')?.textContent).toContain('Euler Ancestral');
    const unipc = Array.from(document.querySelectorAll<HTMLElement>('[role="option"]')).find((option) =>
      option.textContent?.includes('UniPC')
    );
    await interact(() => unipc?.click());
    expect(onChange).toHaveBeenLastCalledWith('unipc');
    expect(input.value).toBe('UniPC');

    await interact(() => input.click());
    await setInputValue(input, 'euler');
    await interact(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' }));
      input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter' }));
    });
    expect(onChange).toHaveBeenLastCalledWith('euler_a');
    expect(input.value).toBe('Euler Ancestral');
  });

  it('clears the controlled search query when reopened', async () => {
    const { input, onInputValueChange } = await renderCombobox(false, true);

    await interact(() => input.click());
    await setInputValue(input, 'DPM');
    expect(document.querySelectorAll('[role="option"]')).toHaveLength(1);

    await interact(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' }));
    });
    expect(onInputValueChange).toHaveBeenLastCalledWith('');
    expect(onInputValueChange).toHaveBeenCalledTimes(2);
    expect(input.getAttribute('aria-expanded')).toBe('false');
    await interact(() => input.click());

    expect(input.value).toBe('');
    expect(document.querySelectorAll('[role="option"]')).toHaveLength(options.length);
  });

  it('scrolls long lists in the themed viewport, following the keyboard highlight and reporting the end', async () => {
    const longOptions = Array.from({ length: 40 }, (_, index) => ({
      label: `Scheduler ${index + 1}`,
      value: `scheduler_${index + 1}`,
    }));
    const onListScrollToBottom = vi.fn();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
    await interact(() => {
      root?.render(
        <I18nextProvider i18n={i18n}>
          <ChakraProvider value={system}>
            <Combobox
              aria-label="Scheduler"
              options={longOptions}
              value="scheduler_1"
              onListScrollToBottom={onListScrollToBottom}
              onValueChange={() => undefined}
            />
          </ChakraProvider>
        </I18nextProvider>
      );
    });
    const input = host.querySelector<HTMLInputElement>('input[role="combobox"]')!;

    await interact(() => input.click());
    const content = document.querySelector<HTMLElement>('[data-scope="combobox"][data-part="content"]')!;
    const viewport = content.querySelector<HTMLElement>('[data-scope="scroll-area"][data-part="viewport"]')!;
    expect(viewport.scrollHeight).toBeGreaterThan(viewport.clientHeight);
    expect(content.scrollHeight).toBe(content.clientHeight);

    for (let step = 0; step < 20; step += 1) {
      await interact(() => {
        input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowDown' }));
      });
    }
    const highlighted = content.querySelector<HTMLElement>('[role="option"][data-highlighted]')!;
    const highlightedRect = highlighted.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();
    expect(highlighted.textContent).toContain('Scheduler 21');
    expect(highlightedRect.top).toBeGreaterThanOrEqual(viewportRect.top);
    expect(highlightedRect.bottom).toBeLessThanOrEqual(viewportRect.bottom);
    expect(onListScrollToBottom).not.toHaveBeenCalled();

    await interact(() => {
      viewport.scrollTop = viewport.scrollHeight;
      viewport.dispatchEvent(new Event('scroll'));
    });
    expect(onListScrollToBottom).toHaveBeenCalled();
  });

  it('honors the disabled state', async () => {
    const { input } = await renderCombobox(true);

    expect(input.disabled).toBe(true);
    await interact(() => input.click());
    expect(input.getAttribute('aria-expanded')).toBe('false');
  });

  describe('under an inside Field label', () => {
    const renderInField = async (field: { disabled?: boolean; error?: string; helpText?: string }) => {
      host = document.createElement('div');
      host.style.width = '240px';
      document.body.append(host);
      root = createRoot(host);
      const onChange = vi.fn();

      await interact(() => {
        root?.render(
          <I18nextProvider i18n={i18n}>
            <ChakraProvider value={system}>
              <Field id="scheduler-field" label="Scheduler" labelPlacement="inside" {...field}>
                <Combobox options={options} value="euler_a" onValueChange={onChange} />
              </Field>
            </ChakraProvider>
          </I18nextProvider>
        );
      });

      const control = host.querySelector<HTMLElement>('[data-scope="combobox"][data-part="control"]')!;
      return {
        control,
        input: control.querySelector<HTMLInputElement>('input[role="combobox"]')!,
        label: control.querySelector<HTMLLabelElement>('label')!,
        onChange,
      };
    };

    it('names the input with the label at the start of its box and describes it with the help text', async () => {
      const { control, input, label } = await renderInField({ helpText: 'Sampler for each step' });

      expect(label.id).toBe('scheduler-field-label');
      expect(label.getBoundingClientRect().right).toBeLessThanOrEqual(input.getBoundingClientRect().left);
      expect(input.value).toBe('Euler Ancestral');
      expect(getComputedStyle(input).textAlign).toBe('end');
      // The box is the control, so the border runs around the label as well as the input.
      expect(getComputedStyle(control).borderTopWidth).toBe('1px');
      expect(getComputedStyle(input).borderTopWidth).toBe('0px');
      const scheduler = page.getByRole('combobox', { exact: true, name: 'Scheduler' });
      await expect.element(scheduler).toHaveAccessibleDescription('Sampler for each step');
    });

    it('opens from a click on the label, then filters and commits as usual', async () => {
      const { input, label, onChange } = await renderInField({});

      await userEvent.click(label);
      await expect.element(page.getByRole('combobox', { exact: true, name: 'Scheduler' })).toHaveFocus();
      expect(input.getAttribute('aria-expanded')).toBe('true');

      await setInputValue(input, 'dpm');
      expect(document.querySelectorAll('[role="option"]')).toHaveLength(1);
      await interact(() => document.querySelector<HTMLElement>('[role="option"]')?.click());
      expect(onChange).toHaveBeenLastCalledWith('dpmpp_2m');
    });

    it('opens from a click anywhere in the box, its inset and the gap before the value included', async () => {
      const { control, input, label } = await renderInField({});
      const box = control.getBoundingClientRect();
      const labelBox = label.getBoundingClientRect();
      const close = () =>
        interact(() => input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Escape' })));

      // The strip inside the left border, before the label's text.
      await userEvent.click(control, { position: { x: 3, y: box.height / 2 } });
      await expect.poll(() => input.getAttribute('aria-expanded')).toBe('true');
      await close();
      await expect.poll(() => input.getAttribute('aria-expanded')).toBe('false');

      // The gap between the label's text and the value.
      const range = document.createRange();
      range.selectNodeContents(label);
      const textRight = range.getBoundingClientRect().right;
      expect(labelBox.right - textRight).toBeGreaterThan(4);
      await userEvent.click(control, { position: { x: (textRight + labelBox.right) / 2 - box.left, y: 3 } });
      await expect.poll(() => input.getAttribute('aria-expanded')).toBe('true');
    });

    it('describes the input with an invalid field error and marks the whole box invalid', async () => {
      const valid = await renderInField({});
      const validBorder = getComputedStyle(valid.control).borderTopColor;
      await interact(() => root?.unmount());
      host?.remove();

      const { control, input } = await renderInField({ error: 'Pick a scheduler' });

      expect(input.getAttribute('aria-invalid')).toBe('true');
      expect(input.getAttribute('aria-errormessage')).toBe('scheduler-field-error');
      await expect
        .element(page.getByRole('combobox', { exact: true, name: 'Scheduler' }))
        .toHaveAccessibleDescription('Pick a scheduler');
      expect(control.hasAttribute('data-invalid')).toBe(true);
      expect(getComputedStyle(control).borderTopColor).not.toBe(validBorder);
    });

    it('dims the whole box once when the field is disabled', async () => {
      const { control, input, label } = await renderInField({ disabled: true });

      expect(input.disabled).toBe(true);
      expect(getComputedStyle(control).opacity).toBe('0.5');
      expect(getComputedStyle(input).opacity).toBe('1');
      await interact(() => label.click());
      expect(input.getAttribute('aria-expanded')).toBe('false');
    });
  });
});
