import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { SliderNumberField } from './SliderNumberField';

const formatScaleValue = (value: number): string => `${value}×`;

const MARKS_WITH_ONE_OFF_TRACK = [
  { label: 'min', value: 0 },
  { label: 'mid', value: 5 },
  { label: 'default', value: 30 },
];

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
    // Select by role=slider; the shared tooltip trigger overwrites thumb data attributes.
    const thumb = hosted.querySelector('[role="slider"]');

    expect(input?.value).toBe('150');
    expect(thumb?.getAttribute('aria-valuenow')).toBe('100');
  });

  it('drops a mark the track cannot place instead of painting it past the end', async () => {
    // Drop out-of-range default marks without changing the input value or falsely marking a bound as default.
    const hosted = await mount(
      <SliderNumberField
        ariaLabel="Guidance"
        marks={MARKS_WITH_ONE_OFF_TRACK}
        max={10}
        min={0}
        numberInputMax={100}
        step={0.5}
        value={30}
        onChange={vi.fn()}
      />
    );

    expect([...hosted.querySelectorAll('[data-part="marker"]')].map((marker) => marker.textContent)).toEqual([
      'min',
      'mid',
    ]);
    expect(hosted.querySelector<HTMLInputElement>('input[aria-label="Guidance"]')?.value).toBe('30');
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
});
