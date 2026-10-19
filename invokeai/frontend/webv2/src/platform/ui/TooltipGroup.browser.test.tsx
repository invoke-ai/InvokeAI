import type { KeyboardEvent } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { userEvent } from 'vitest/browser';

import { TooltipGroup } from './Tooltip';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const POLL = { timeout: 5000 } as const;
const VALUES = ['alpha', 'beta', 'gamma'] as const;
// Gamma has nothing to say.
const isEnabled = (value: string) => value !== 'gamma';
const getTriggerId = (value: string) => `group-test-${value}`;

/** Roving focus as tab strips do it: the key handler moves focus itself, inside the key event. */
const moveFocus = (event: KeyboardEvent<HTMLButtonElement>) => {
  if (event.key !== 'ArrowRight') {
    return;
  }

  event.preventDefault();
  (event.currentTarget.nextElementSibling as HTMLElement | null)?.focus();
};

const Group = () => (
  <TooltipGroup.Root content="Shared tip" getTriggerId={getTriggerId} isEnabled={isEnabled}>
    <div role="toolbar">
      {VALUES.map((value) => (
        <TooltipGroup.Trigger key={value} value={value}>
          <button id={getTriggerId(value)} tabIndex={value === 'alpha' ? 0 : -1} type="button" onKeyDown={moveFocus}>
            {value}
          </button>
        </TooltipGroup.Trigger>
      ))}
    </div>
  </TooltipGroup.Root>
);

let host: HTMLDivElement;
let root: Root;
const trigger = (value: string) => document.getElementById(getTriggerId(value))!;
const openTip = () =>
  document.querySelector<HTMLElement>('[data-scope="tooltip"][data-part="content"][data-state="open"]');
const describedTriggers = () =>
  VALUES.filter((value) => {
    const describedBy = trigger(value).getAttribute('aria-describedby');

    return describedBy !== null && describedBy === openTip()?.id;
  });

beforeEach(async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <Group />
      </ChakraProvider>
    )
  );
});

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

describe('TooltipGroup', () => {
  it('hands the tip from trigger to trigger under roving focus, describing only the one it is on', async () => {
    await act(() => userEvent.tab());
    expect(document.activeElement).toBe(trigger('alpha'));
    await expect.poll(() => openTip()?.textContent, POLL).toBe('Shared tip');
    await expect.poll(describedTriggers, POLL).toEqual(['alpha']);

    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(document.activeElement).toBe(trigger('beta'));
    await expect.poll(describedTriggers, POLL).toEqual(['beta']);
    expect(openTip()?.textContent).toBe('Shared tip');
  });

  it('shows nothing on a trigger that is not enabled', async () => {
    await act(() => userEvent.tab());
    await expect.poll(() => openTip()?.textContent, POLL).toBe('Shared tip');

    await act(() => userEvent.keyboard('{ArrowRight}'));
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(document.activeElement).toBe(trigger('gamma'));
    await expect.poll(openTip, POLL).toBeNull();
    expect(describedTriggers()).toEqual([]);
  });
});
