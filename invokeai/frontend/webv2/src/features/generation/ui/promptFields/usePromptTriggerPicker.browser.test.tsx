import { ChakraProvider } from '@chakra-ui/react';
import { usePromptTriggerPicker } from '@features/generation/ui/promptFields/usePromptTriggerPicker';
import { system } from '@theme/system';
import { act, useRef } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const PickerProbe = () => {
  const anchorRef = useRef<HTMLButtonElement | null>(null);
  const picker = usePromptTriggerPicker({ insert: vi.fn() });

  return (
    <>
      <button ref={anchorRef} onClick={() => picker.open(anchorRef.current!)}>
        Open
      </button>
      {picker.dismissElement}
      {picker.isOpen ? <div role="dialog">Picker</div> : null}
    </>
  );
};

const render = async () => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        <PickerProbe />
      </ChakraProvider>
    );
  });

  await act(async () => {
    await userEvent.click(host!.querySelector('button')!);
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('the button-anchored prompt trigger picker', () => {
  it('closes on a captured element scroll', async () => {
    await render();
    expect(host!.querySelector('[role="dialog"]')).not.toBeNull();

    await act(() => {
      host!.dispatchEvent(new Event('scroll'));
    });

    expect(host!.querySelector('[role="dialog"]')).toBeNull();
  });

  it('closes on viewport resize', async () => {
    await render();

    await act(() => {
      window.dispatchEvent(new Event('resize'));
    });

    expect(host!.querySelector('[role="dialog"]')).toBeNull();
  });
});
