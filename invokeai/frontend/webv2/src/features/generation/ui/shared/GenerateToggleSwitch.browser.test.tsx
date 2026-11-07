/* oxlint-disable react-perf/jsx-no-new-function-as-prop, react-perf/jsx-no-jsx-as-prop */
import { Box, ChakraProvider, Input } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { GenerateToggleSwitch } from './GenerateToggleSwitch';

const LONG_LABEL = 'Keep the reference image key and value cache between denoising steps on this device';
const HELP = 'Reuses each reference image encoding across steps, which is faster but holds more memory.';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

/** Commits land back on the row, as the Generate draft does. */
const Harness = ({
  disabled,
  initial,
  label = LONG_LABEL,
  onChange,
  paired,
  preview,
}: {
  disabled?: boolean;
  initial: boolean;
  label?: string;
  onChange: (checked: boolean) => void;
  paired?: boolean;
  preview?: boolean;
}) => {
  const [checked, setChecked] = useState(initial);

  return (
    <GenerateToggleSwitch
      checked={checked}
      disabled={disabled}
      helpText={HELP}
      label={label}
      preview={preview ? <Box data-testid="preview" h="3" w="12" /> : undefined}
      onCheckedChange={(next) => {
        onChange(next);
        setChecked(next);
      }}
    >
      {paired ? <Input aria-label="Paired value" /> : null}
    </GenerateToggleSwitch>
  );
};

const render = async (
  props: { disabled?: boolean; initial?: boolean; label?: string; paired?: boolean; preview?: boolean } = {}
) => {
  const onChange = vi.fn();

  host = document.createElement('div');
  // The narrowest Generate widget.
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <Harness initial={props.initial ?? false} onChange={onChange} {...props} />
      </ChakraProvider>
    )
  );

  return onChange;
};

const toggleInput = () => host!.querySelector<HTMLInputElement>('input[type="checkbox"]')!;
const visibleLabel = () => host!.querySelector<HTMLElement>('[data-scope="switch"][data-part="label"]')!;
const control = () => host!.querySelector<HTMLElement>('[data-scope="switch"][data-part="control"]')!;

describe('GenerateToggleSwitch', () => {
  it('toggles from its visible label, which is also its accessible name', async () => {
    const onChange = await render();

    expect(document.getElementById(toggleInput().getAttribute('aria-labelledby')!)?.textContent).toBe(LONG_LABEL);
    // Controls keep the arrow cursor, as everywhere in the app.
    expect(getComputedStyle(visibleLabel()).cursor).toBe('default');

    await userEvent.click(visibleLabel());
    expect(onChange).toHaveBeenLastCalledWith(true);
    expect(toggleInput().checked).toBe(true);

    await userEvent.click(visibleLabel());
    expect(onChange).toHaveBeenLastCalledWith(false);
  });

  it('describes the switch with its help text, which does not toggle it', async () => {
    const onChange = await render();
    const help = document.getElementById(toggleInput().getAttribute('aria-describedby')!);

    expect(help?.textContent).toBe(HELP);
    await userEvent.click(help!);
    expect(onChange).not.toHaveBeenCalled();
  });

  it('ignores a click on a disabled label, and shows the label disabled', async () => {
    const onChange = await render({ disabled: true });

    await userEvent.click(visibleLabel(), { force: true });

    expect(onChange).not.toHaveBeenCalled();
    expect(toggleInput().checked).toBe(false);
    expect(getComputedStyle(visibleLabel()).cursor).toBe('not-allowed');
    expect(Number(getComputedStyle(visibleLabel()).opacity)).toBeLessThan(1);
  });

  it('wraps a long label in a narrow widget, keeping the switch whole at the trailing edge of its first line', async () => {
    await render();

    const label = visibleLabel().getBoundingClientRect();
    const switchBox = control().getBoundingClientRect();
    const row = host!.getBoundingClientRect();
    const lineHeight = Number.parseFloat(getComputedStyle(visibleLabel()).lineHeight);

    // Wrapped onto several lines rather than truncated or clipped.
    expect(label.height).toBeGreaterThan(lineHeight * 2);
    expect(visibleLabel().scrollWidth).toBeLessThanOrEqual(visibleLabel().clientWidth);
    expect(label.right).toBeLessThanOrEqual(switchBox.left);
    // Pinned to the trailing edge, the same size as beside a label that fits on one line.
    expect(switchBox.right).toBeCloseTo(row.right, 0);
    await act(() => root?.unmount());
    host?.remove();
    await render({ label: 'On' });
    const reference = control().getBoundingClientRect();

    expect(reference.width).toBeGreaterThan(0);
    expect(switchBox.width).toBe(reference.width);
    expect(switchBox.height).toBe(reference.height);
    // Level with the label's first line, not the middle of the wrapped block.
    expect(switchBox.top + switchBox.height / 2).toBeCloseTo(label.top + lineHeight / 2, 0);
  });

  it('sends a label click to the switch, never to the control it pairs with', async () => {
    const onChange = await render({ paired: true });
    const paired = host!.querySelector<HTMLInputElement>('input[aria-label="Paired value"]')!;

    await userEvent.click(visibleLabel());

    expect(onChange).toHaveBeenCalledWith(true);
    expect(document.activeElement).not.toBe(paired);

    await userEvent.click(paired);
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it('keeps a click on the preview from toggling, and levels the preview with the switch', async () => {
    const onChange = await render({ preview: true });
    const preview = host!.querySelector<HTMLElement>('[data-testid="preview"]')!;
    const previewBox = preview.getBoundingClientRect();
    const switchBox = control().getBoundingClientRect();

    expect(previewBox.right).toBeLessThanOrEqual(switchBox.left);
    expect(previewBox.top + previewBox.height / 2).toBeCloseTo(switchBox.top + switchBox.height / 2, 0);

    await userEvent.click(preview);
    expect(onChange).not.toHaveBeenCalled();
    expect(toggleInput().checked).toBe(false);
  });
});
