/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
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
  { label: 'UniPC', value: 'unipc' },
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

const Harness = ({ disabled = false, onChange }: { disabled?: boolean; onChange: (value: string) => void }) => {
  const [value, setValue] = useState('euler_a');

  return (
    <Combobox
      aria-label="Scheduler"
      disabled={disabled}
      options={options}
      value={value}
      onValueChange={(nextValue) => {
        setValue(nextValue);
        onChange(nextValue);
      }}
    />
  );
};

const renderCombobox = async (disabled = false) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onChange = vi.fn();

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness disabled={disabled} onChange={onChange} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  return { input: host.querySelector<HTMLInputElement>('input[role="combobox"]')!, onChange };
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

  it('honors the disabled state', async () => {
    const { input } = await renderCombobox(true);

    expect(input.disabled).toBe(true);
    await interact(() => input.click());
    expect(input.getAttribute('aria-expanded')).toBe('false');
  });
});
