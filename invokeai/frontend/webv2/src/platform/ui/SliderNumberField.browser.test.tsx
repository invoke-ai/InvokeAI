import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SliderNumberField } from './SliderNumberField';

const formatScaleValue = (value: number): string => `${value}×`;

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const mount = async (element: React.ReactElement): Promise<HTMLDivElement> => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(<ChakraProvider value={system}>{element}</ChakraProvider>);
  });

  return host;
};

const settle = () =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 50);
    });
  });

describe('SliderNumberField', () => {
  it('clamps the slider thumb to max while the input keeps showing a typed value above it', async () => {
    const hosted = await mount(
      <SliderNumberField
        ariaLabel="Steps"
        max={100}
        min={1}
        numberInputMax={Number.MAX_SAFE_INTEGER}
        step={1}
        value={150}
        onChange={vi.fn()}
      />
    );

    const input = hosted.querySelector<HTMLInputElement>('input[aria-label="Steps"]');
    // The thumb is also the tooltip trigger (`asChild`), so the tooltip's own
    // `data-part`/`data-scope` win on the shared node — `role="slider"` is the
    // stable selector, not `[data-part="thumb"]`.
    const thumb = hosted.querySelector('[role="slider"]');

    expect(input?.value).toBe('150');
    expect(thumb?.getAttribute('aria-valuenow')).toBe('100');
  });

  it('shows the formatted value on the thumb tooltip while it is focused', async () => {
    const hosted = await mount(
      <SliderNumberField
        ariaLabel="Scale"
        formatValue={formatScaleValue}
        max={16}
        min={1}
        step={0.5}
        value={4}
        onChange={vi.fn()}
      />
    );

    await act(async () => {
      await userEvent.tab();
    });
    await settle();

    const thumb = hosted.querySelector('[role="slider"]');

    expect(document.activeElement).toBe(thumb);
    expect(thumb?.getAttribute('data-state')).toBe('open');
    expect(document.body.textContent).toContain('4×');
  });

  it('renders the stepper only when showStepper is set', async () => {
    const withoutStepper = await mount(
      <SliderNumberField ariaLabel="Creativity" max={10} min={0} step={1} value={5} onChange={vi.fn()} />
    );

    // `[data-part="control"]` is ambiguous — the slider has its own "control"
    // wrapper — so scope the query to the number-input's zag-js scope.
    expect(withoutStepper.querySelector('[data-scope="number-input"][data-part="control"]')).toBeNull();

    await act(() => root?.unmount());
    host?.remove();

    const withStepper = await mount(
      <SliderNumberField ariaLabel="Creativity" max={10} min={0} showStepper step={1} value={5} onChange={vi.fn()} />
    );

    expect(withStepper.querySelector('[data-scope="number-input"][data-part="control"]')).not.toBeNull();
  });

  it('shows the reset affordance only while the value differs from the default, and resets on click', async () => {
    const onChange = vi.fn();
    const hosted = await mount(
      <SliderNumberField
        ariaLabel="Steps"
        defaultValue={30}
        max={100}
        min={1}
        resetLabel="Use model default"
        step={1}
        value={45}
        onChange={onChange}
      />
    );

    const resetButton = hosted.querySelector<HTMLButtonElement>('button[aria-label="Use model default"]');

    expect(resetButton).not.toBeNull();

    await act(() => {
      resetButton?.click();
    });

    expect(onChange).toHaveBeenCalledWith(30);

    await act(() => root?.unmount());
    host?.remove();

    const atDefault = await mount(
      <SliderNumberField
        ariaLabel="Steps"
        defaultValue={30}
        max={100}
        min={1}
        resetLabel="Use model default"
        step={1}
        value={30}
        onChange={vi.fn()}
      />
    );

    expect(atDefault.querySelector('button[aria-label="Use model default"]')).toBeNull();
  });
});
