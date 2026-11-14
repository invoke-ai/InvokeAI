import { ChakraProvider, createListCollection } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { Field } from './Field';
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
const SCHEDULER_ITEMS = [
  { label: 'Euler', value: 'euler' },
  { label: 'DPM++ 2M Karras with an unusually long descriptive name', value: 'dpmpp_2m_k' },
];
const EULER_VALUE = ['euler'];
const LONG_VALUE = ['dpmpp_2m_k'];
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

  it('opens with a header-less group without a divider above it', async () => {
    // The header-less group comes first: a divider belongs between groups, never above the first option.
    const items = [DETECTOR_ITEMS[2]!, DETECTOR_ITEMS[0]!, DETECTOR_ITEMS[1]!];
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Select
            aria-label="Detection"
            collection={createListCollection({ items })}
            groupBy={getDetectorGroup}
            renderGroupLabel={renderDetectorGroupLabel}
            value={DETECTOR_VALUE}
          />
        </ChakraProvider>
      );
    });
    await act(() => trigger().click());
    await expect.poll(() => document.querySelectorAll('[role="option"]').length).toBe(3);

    const listbox = document.querySelector<HTMLElement>('[role="listbox"]')!;
    expect(listbox.querySelectorAll('[role="separator"]')).toHaveLength(0);
    expect(listbox.querySelector('[role="option"]')?.textContent).toContain('None');
  });
});

describe('Select under an inside Field label', () => {
  let host: HTMLDivElement;
  let root: Root;
  const trigger = () => host.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]')!;
  const insideLabel = () => trigger().querySelector<HTMLElement>('#scheduler-field-label')!;
  const valueText = () => trigger().querySelector<HTMLElement>('[data-part="value-text"]')!;
  const mount = (
    field: { disabled?: boolean; error?: string; helpText?: string },
    value: string[],
    onValueChange = vi.fn()
  ) =>
    act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field id="scheduler-field" label="Scheduler" labelPlacement="inside" {...field}>
            <Select
              collection={createListCollection({ items: SCHEDULER_ITEMS })}
              value={value}
              onValueChange={onValueChange}
            />
          </Field>
        </ChakraProvider>
      );
    });

  beforeEach(() => {
    host = document.createElement('div');
    host.style.width = '220px';
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(async () => {
    await act(() => root.unmount());
    host.remove();
  });

  it('shows the label at the start of its box, names the trigger with it and describes it with the help text', async () => {
    await mount({ helpText: 'Sampler for each step' }, EULER_VALUE);

    // One box: no label above it, the label and the value side by side inside the trigger.
    expect(host.querySelector('label')).toBeNull();
    expect(insideLabel().textContent).toBe('Scheduler');
    expect(insideLabel().getBoundingClientRect().right).toBeLessThanOrEqual(valueText().getBoundingClientRect().left);
    expect(getComputedStyle(valueText()).textAlign).toBe('end');
    expect(valueText().textContent).toBe('Euler');
    await expect
      .element(page.getByRole('combobox', { exact: true, name: 'Scheduler' }))
      .toHaveAccessibleDescription('Sampler for each step');
    // The help line sits below the box.
    const help = host.querySelector<HTMLElement>('#scheduler-field-help')!;
    expect(help.getBoundingClientRect().top).toBeGreaterThanOrEqual(trigger().getBoundingClientRect().bottom);
  });

  it('opens from a click on the label and commits a choice', async () => {
    const onValueChange = vi.fn();
    await mount({}, EULER_VALUE, onValueChange);

    await userEvent.click(insideLabel());
    await expect.element(page.getByRole('listbox', { exact: true, name: 'Scheduler' })).toBeVisible();
    await userEvent.click(page.getByRole('option', { name: /^DPM\+\+ 2M Karras/ }));

    expect(onValueChange).toHaveBeenLastCalledWith(expect.objectContaining({ value: ['dpmpp_2m_k'] }));
  });

  it('truncates a long value before its label, short of the chevron', async () => {
    await mount({}, LONG_VALUE);

    const label = insideLabel();
    const value = valueText();
    expect(label.scrollWidth).toBeLessThanOrEqual(label.clientWidth);
    expect(value.scrollWidth).toBeGreaterThan(value.clientWidth);
    expect(getComputedStyle(value).textOverflow).toBe('ellipsis');
    const indicator = trigger().querySelector<HTMLElement>('[data-part="indicator"]')!;
    expect(value.getBoundingClientRect().right).toBeLessThanOrEqual(indicator.getBoundingClientRect().left);
    expect(indicator.getBoundingClientRect().right).toBeLessThanOrEqual(trigger().getBoundingClientRect().right);
  });

  it('caps a label longer than half the box so the value keeps the rest', async () => {
    await act(() => {
      root.render(
        <ChakraProvider value={system}>
          <Field label="A very long localized scheduler label that cannot fit" labelPlacement="inside">
            <Select collection={createListCollection({ items: SCHEDULER_ITEMS })} value={EULER_VALUE} />
          </Field>
        </ChakraProvider>
      );
    });

    const label = trigger().querySelector<HTMLElement>('span[id]')!;
    expect(label.scrollWidth).toBeGreaterThan(label.clientWidth);
    expect(label.getBoundingClientRect().width).toBeLessThanOrEqual(trigger().getBoundingClientRect().width / 2);
    expect(valueText().scrollWidth).toBeLessThanOrEqual(valueText().clientWidth);
  });

  it('dims its chevron once with the rest of the trigger when the field is disabled', async () => {
    const indicatorColor = () =>
      getComputedStyle(trigger().querySelector<HTMLElement>('[data-part="indicator"]')!).color;
    await mount({}, EULER_VALUE);
    const restingColor = indicatorColor();

    await mount({ disabled: true }, EULER_VALUE);
    expect(Number(getComputedStyle(trigger()).opacity)).toBeLessThan(1);
    expect(indicatorColor()).toBe(restingColor);
  });

  it('takes the field error as its invalid state and description, and the field disabled state', async () => {
    await mount({ error: 'Pick a scheduler' }, EULER_VALUE);

    expect(trigger().getAttribute('aria-invalid')).toBe('true');
    await expect
      .element(page.getByRole('combobox', { exact: true, name: 'Scheduler' }))
      .toHaveAccessibleDescription('Pick a scheduler');

    await mount({ disabled: true }, EULER_VALUE);
    expect(trigger().disabled).toBe(true);
    await userEvent.click(insideLabel(), { force: true });
    expect(document.querySelector('[role="listbox"]')).toBeNull();
  });
});
