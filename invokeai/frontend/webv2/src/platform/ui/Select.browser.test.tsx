import { ChakraProvider, createListCollection } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { Select } from './Select';

const RATIO_ITEMS = [
  { label: '16:9', value: '16:9' },
  { label: '1:1', value: '1:1' },
];
const RATIO_VALUE = ['16:9'];
const DETECTOR_ITEMS = [
  { group: 'detectors', label: 'Canny', value: 'canny' },
  { group: 'detectors', label: 'MLSD', value: 'mlsd' },
  { group: 'none', label: 'None: image is already an edge map', value: 'none' },
];
const DETECTOR_VALUE = ['canny'];
const getDetectorGroup = (item: { group: string }) => item.group;
const renderDetectorGroupLabel = (group: string) => (group === 'detectors' ? 'Detectors' : null);

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe('Select', () => {
  let host: HTMLDivElement;
  let root: Root;
  const mount = (items: { label: string; value: string }[]) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select aria-label="Scheduler" collection={createListCollection({ items })} />
        </ChakraProvider>
      );
    });
  const trigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]')!;

  beforeEach(() => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('shows the selected label by default and lets a caller name the value instead', async () => {
    // An omitted trigger-text override must fall back to the selection, not an empty display.
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Aspect ratio"
            collection={createListCollection({ items: RATIO_ITEMS })}
            value={RATIO_VALUE}
          />
        </ChakraProvider>
      );
    });
    expect(trigger().textContent).toContain('16:9');

    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Aspect ratio"
            collection={createListCollection({ items: RATIO_ITEMS })}
            value={RATIO_VALUE}
            valueText="From the first frame"
          />
        </ChakraProvider>
      );
    });
    expect(trigger().textContent).toContain('From the first frame');
    expect(trigger().textContent).not.toContain('16:9');
  });

  it('disables the trigger while the collection is empty, and opens once it has items', async () => {
    await mount([]);
    expect(trigger().disabled).toBe(true);
    await act(() => trigger().click());
    expect(document.querySelector('[role="listbox"]')).toBeNull();

    await mount([{ label: 'Euler', value: 'euler' }]);
    expect(trigger().disabled).toBe(false);
    await act(() => trigger().click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(1);
  });

  it('puts a header-less group under a divider and moves the keyboard across it', async () => {
    const onValueChange = vi.fn();
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Detection"
            collection={createListCollection({ items: DETECTOR_ITEMS })}
            groupBy={getDetectorGroup}
            renderGroupLabel={renderDetectorGroupLabel}
            value={DETECTOR_VALUE}
            onValueChange={onValueChange}
          />
        </ChakraProvider>
      );
    });
    await act(() => trigger().click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(3);

    const labels = Array.from(document.querySelectorAll('[data-scope="select"][data-part="item-group-label"]'));
    expect(labels.map((label) => label.textContent)).toEqual(['Detectors']);
    const listbox = document.querySelector<HTMLElement>('[role="listbox"]')!;
    const separators = listbox.querySelectorAll('[role="separator"]');
    expect(separators).toHaveLength(1);
    // The divider sits between the labelled group and the trailing option, hidden from assistive technology.
    expect(separators[0]!.getAttribute('aria-hidden')).toBe('true');
    expect(separators[0]!.previousElementSibling?.textContent).toContain('Detectors');
    expect(separators[0]!.nextElementSibling?.getAttribute('role')).toBe('option');
    expect(separators[0]!.nextElementSibling?.textContent).toContain('None');

    await act(() => userEvent.keyboard('{ArrowDown}{ArrowDown}{Enter}'));
    expect(onValueChange).toHaveBeenLastCalledWith(expect.objectContaining({ value: ['none'] }));
  });
});
