import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { LoadError } from './LoadError';

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const mount = async (onRetry: () => unknown) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <LoadError
          detail="Database is locked"
          isRetrying={false}
          layout="inline"
          message="Could not load"
          onRetry={onRetry}
        />
      </ChakraProvider>
    )
  );
  return host.querySelector<HTMLButtonElement>('[role="alert"] button')!;
};

describe('LoadError', () => {
  it('starts one attempt per press, however fast the clicks, and keeps focus while it runs', async () => {
    let finish!: () => void;
    const onRetry = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        })
    );
    const retry = await mount(onRetry);

    expect(host!.querySelector('[role="alert"]')?.textContent).toContain('Database is locked');

    await act(() => userEvent.dblClick(retry));

    expect(onRetry).toHaveBeenCalledOnce();
    expect(retry.getAttribute('aria-busy')).toBe('true');
    expect(document.activeElement).toBe(retry);

    await act(async () => {
      finish();
      await Promise.resolve();
    });
    expect(retry.hasAttribute('aria-busy')).toBe(false);

    await act(() => userEvent.click(retry));
    expect(onRetry).toHaveBeenCalledTimes(2);
  });
});
