import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { CircleIcon, SquareIcon, StarIcon } from 'lucide-react';
import { act, useCallback, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { SegmentedControl, type SegmentedControlOption } from './SegmentedControl';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ICON_OPTIONS: readonly SegmentedControlOption[] = [
  { icon: SquareIcon, label: 'Square', value: 'square' },
  { disabled: true, icon: CircleIcon, label: 'Circle', value: 'circle' },
  { icon: StarIcon, label: 'Star', value: 'star' },
];

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const mount = async (options: readonly SegmentedControlOption[], onChange = vi.fn()) => {
  const Harness = () => {
    const [value, setValue] = useState(options[0]!.value);
    const select = useCallback((next: string) => {
      onChange(next);
      setValue(next);
    }, []);
    return <SegmentedControl ariaLabel="Kind" options={options} value={value} onChange={select} />;
  };
  host = document.createElement('div');
  host.style.width = '160px';
  // Room on both sides, so the tip centres on its segment instead of shifting off the viewport edge.
  host.style.margin = '80px 200px';
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root!.render(
      <ChakraProvider value={system}>
        <Harness />
      </ChakraProvider>
    )
  );
  return onChange;
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('SegmentedControl icon options', () => {
  it('names each icon segment, shows its label as a tooltip on focus, and steps over disabled ones', async () => {
    const onChange = await mount(ICON_OPTIONS);
    const group = page.getByRole('radiogroup', { name: 'Kind' });
    const square = group.getByRole('radio', { exact: true, name: 'Square' });
    await expect.element(square).toBeChecked();
    // The label is only the accessible name: the segment shows the icon alone.
    expect(host!.querySelector('[data-part="item-text"]')!.getBoundingClientRect().width).toBeLessThanOrEqual(1);

    await userEvent.keyboard('{Tab}');
    await expect.element(square).toHaveFocus();
    const tooltip = page.getByRole('tooltip');
    await expect.element(tooltip).toHaveTextContent('Square');
    // Anchored to its own segment: the tip sits over Square, not at the page origin.
    const item = square.element().closest('[data-part="item"]')!;
    // The tip leaves the segment's own checked state in place.
    expect(item.getAttribute('data-state')).toBe('checked');
    const segment = item.getBoundingClientRect();
    await expect
      .poll(() => {
        const tip = tooltip.element().getBoundingClientRect();
        return Math.abs(tip.left + tip.width / 2 - (segment.left + segment.width / 2));
      })
      .toBeLessThan(4);

    await userEvent.keyboard('{ArrowRight}');

    expect(onChange).toHaveBeenLastCalledWith('star');
    await expect.element(group.getByRole('radio', { exact: true, name: 'Star' })).toBeChecked();
    await expect.element(group.getByRole('radio', { exact: true, name: 'Circle' })).toBeDisabled();
    await expect.element(page.getByRole('tooltip')).toHaveTextContent('Star');
  });

  it('keeps text segments as visible labels without tooltips', async () => {
    await mount([
      { label: 'Linear', value: 'linear' },
      { label: 'Radial', value: 'radial' },
    ]);
    const linear = page.getByRole('radio', { exact: true, name: 'Linear' });
    await expect.element(page.getByText('Linear')).toBeVisible();

    await userEvent.keyboard('{Tab}');

    await expect.element(linear).toHaveFocus();
    expect(document.querySelector('[role="tooltip"]')).toBeNull();
  });
});
