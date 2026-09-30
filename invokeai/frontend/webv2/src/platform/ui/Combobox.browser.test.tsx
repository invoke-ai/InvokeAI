/* oxlint-disable react-perf/jsx-no-new-array-as-prop, react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { Combobox, type ComboboxOption } from './Combobox';

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
          noSchedulersFound: 'No schedulers found',
          openSelector: 'Open selector',
          searchSchedulers: 'Search schedulers…',
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
    expect(document.body.textContent).toContain('No schedulers found');
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
});
