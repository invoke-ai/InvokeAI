/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ColorPicker } from './ColorPicker';

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        common: {
          colorPicker: {
            format: 'Color format',
            opacity: 'Opacity (%)',
            sampleFromCanvas: 'Pick a color from the canvas',
            sampleFromScreen: 'Pick a color from the screen',
          },
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

interface HarnessProps {
  initialValue: string;
  swatches?: boolean | readonly string[];
  withAlpha?: boolean;
  onChange: (value: string) => void;
  onChangeEnd: (value: string) => void;
}

const Harness = ({ initialValue, onChange, onChangeEnd, swatches, withAlpha }: HarnessProps) => {
  const [value, setValue] = useState(initialValue);

  return (
    <ColorPicker
      aria-label="Fill color"
      swatches={swatches}
      value={value}
      withAlpha={withAlpha}
      onValueChange={(next) => {
        setValue(next);
        onChange(next);
      }}
      onValueChangeEnd={onChangeEnd}
    />
  );
};

const renderPicker = async (props: Omit<HarnessProps, 'onChange' | 'onChangeEnd'>) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  const onChange = vi.fn();
  const onChangeEnd = vi.fn();

  await interact(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness {...props} onChange={onChange} onChangeEnd={onChangeEnd} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });

  const trigger = host.querySelector<HTMLButtonElement>('[data-scope="color-picker"][data-part="trigger"]')!;

  return { onChange, onChangeEnd, trigger };
};

const openPicker = async (trigger: HTMLButtonElement) => {
  await interact(() => trigger.click());
};

const querySwatchTriggers = (): HTMLButtonElement[] =>
  Array.from(document.querySelectorAll<HTMLButtonElement>('[data-part="swatch-trigger"]'));

const queryChannelInputs = (): HTMLInputElement[] =>
  Array.from(document.querySelectorAll<HTMLInputElement>('[data-part="channel-input"]'));

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(async () => {
  await interact(() => root?.unmount());
  document.querySelectorAll('[data-scope="color-picker"][data-part="positioner"]').forEach((el) => el.remove());
  document.querySelectorAll('[data-scope="menu"][data-part="positioner"]').forEach((el) => el.remove());
  host?.remove();
  host = null;
  root = null;
  window.localStorage.clear();
});

describe('ColorPicker', () => {
  it('opens a popover with an area, hue slider, channel inputs, and swatches', async () => {
    const { trigger } = await renderPicker({ initialValue: '#ff0000' });
    await openPicker(trigger);

    expect(document.querySelector('[data-part="area"]')).not.toBeNull();
    expect(document.querySelector('[data-part="channel-slider"][data-channel="hue"]')).not.toBeNull();
    expect(queryChannelInputs().length).toBeGreaterThan(0);
    expect(querySwatchTriggers().length).toBeGreaterThan(0);
  });

  it('emits both a change and a commit when a swatch is picked', async () => {
    // Zag's SwatchTrigger only reports a value change; without the component's
    // explicit commit, every undo-recording consumer would drop swatch picks.
    const { onChange, onChangeEnd, trigger } = await renderPicker({
      initialValue: '#000000',
      swatches: ['#ff0000', '#00ff00'],
    });
    await openPicker(trigger);

    const [, green] = querySwatchTriggers();
    await interact(() => green?.click());

    expect(onChange).toHaveBeenLastCalledWith('#00ff00');
    expect(onChangeEnd).toHaveBeenLastCalledWith('#00ff00');
  });

  it('emits 6-digit hex by default and 8-digit under withAlpha', async () => {
    const opaque = await renderPicker({ initialValue: '#000000', swatches: ['#ff0000'] });
    await openPicker(opaque.trigger);
    await interact(() => querySwatchTriggers()[0]?.click());
    expect(opaque.onChange).toHaveBeenLastCalledWith('#ff0000');

    await interact(() => root?.unmount());
    host?.remove();

    const alpha = await renderPicker({ initialValue: '#000000ff', swatches: ['#ff0000'], withAlpha: true });
    await openPicker(alpha.trigger);
    await interact(() => querySwatchTriggers()[0]?.click());
    expect(alpha.onChange).toHaveBeenLastCalledWith('#ff0000ff');
  });

  it('takes the alpha a swatch carries, so recents restore their transparency', async () => {
    const { onChange, trigger } = await renderPicker({
      initialValue: '#000000ff',
      swatches: ['#ff000080'],
      withAlpha: true,
    });
    await openPicker(trigger);
    await interact(() => querySwatchTriggers()[0]?.click());

    expect(onChange).toHaveBeenLastCalledWith('#ff000080');
  });

  it('shows an alpha slider only under withAlpha', async () => {
    const opaque = await renderPicker({ initialValue: '#ff0000' });
    await openPicker(opaque.trigger);
    expect(document.querySelector('[data-part="channel-slider"][data-channel="alpha"]')).toBeNull();

    await interact(() => root?.unmount());
    host?.remove();

    const alpha = await renderPicker({ initialValue: '#ff0000ff', withAlpha: true });
    await openPicker(alpha.trigger);
    expect(document.querySelector('[data-part="channel-slider"][data-channel="alpha"]')).not.toBeNull();
  });

  it('cycles formats through HEX/RGB/HSL/HSB without changing the value', async () => {
    const { onChange, trigger } = await renderPicker({ initialValue: '#ff0000', swatches: false });
    await openPicker(trigger);

    const formatTrigger = () => document.querySelector<HTMLButtonElement>('[aria-label="Color format"]')!;

    // HEX renders a single text field.
    expect(formatTrigger().textContent).toBe('HEX');
    expect(queryChannelInputs()).toHaveLength(1);

    // RGB renders one field per channel.
    await interact(() => formatTrigger().click());
    expect(formatTrigger().textContent).toBe('RGB');
    expect(queryChannelInputs()).toHaveLength(3);

    await interact(() => formatTrigger().click());
    expect(formatTrigger().textContent).toBe('HSL');
    expect(queryChannelInputs()).toHaveLength(3);

    await interact(() => formatTrigger().click());
    expect(formatTrigger().textContent).toBe('HSB');

    // Back to the start, and cycling never emitted a value change.
    await interact(() => formatTrigger().click());
    expect(formatTrigger().textContent).toBe('HEX');
    expect(onChange).not.toHaveBeenCalled();
  });

  it('edits opacity as a whole percentage rather than a unit float', async () => {
    // Zag's own alpha channel input renders `0.501`; the picker replaces it.
    const { onChange, trigger } = await renderPicker({
      initialValue: '#ff000080',
      swatches: false,
      withAlpha: true,
    });
    await openPicker(trigger);

    const opacity = document.querySelector<HTMLInputElement>('[aria-label="Opacity (%)"]')!;
    expect(opacity.value).toBe('50');

    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    await interact(() => {
      setValue?.call(opacity, '25');
      opacity.dispatchEvent(new InputEvent('input', { bubbles: true, data: '25', inputType: 'insertText' }));
    });

    expect(onChange).toHaveBeenLastCalledWith('#ff000040');
  });

  it('offers the opacity field only under withAlpha', async () => {
    const opaque = await renderPicker({ initialValue: '#ff0000', swatches: false });
    await openPicker(opaque.trigger);
    expect(document.querySelector('[aria-label="Opacity (%)"]')).toBeNull();

    await interact(() => root?.unmount());
    host?.remove();

    const alpha = await renderPicker({ initialValue: '#ff0000ff', swatches: false, withAlpha: true });
    await openPicker(alpha.trigger);
    expect(document.querySelector('[aria-label="Opacity (%)"]')).not.toBeNull();
  });

  it('records committed colors as recents and offers them alongside the defaults', async () => {
    const first = await renderPicker({ initialValue: '#000000', swatches: ['#123456'] });
    await openPicker(first.trigger);
    const defaultCount = querySwatchTriggers().length;
    await interact(() => querySwatchTriggers()[0]?.click());

    await interact(() => root?.unmount());
    host?.remove();

    // A picker using the default palette should now surface the committed color.
    const second = await renderPicker({ initialValue: '#000000' });
    await openPicker(second.trigger);
    const values = querySwatchTriggers().map((swatch) => swatch.getAttribute('data-value'));

    expect(defaultCount).toBe(1);
    expect(values).toContain('#123456');
  });
});
