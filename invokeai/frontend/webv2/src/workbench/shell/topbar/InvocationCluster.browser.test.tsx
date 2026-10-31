import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { accountLifecycle } from '@platform/state/accountLifecycle';
import { Button } from '@platform/ui/Button';
import { system } from '@theme/system';
import { patchWorkbenchPreferences } from '@workbench/settings/store';
import { createInitialWorkbenchState } from '@workbench/workbenchState.testing';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { page } from 'vitest/browser';

vi.mock('@workbench/projects/api', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  setClientStateValue: async () => {},
}));
vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProjectSelector: (selector: (project: Project) => unknown) =>
    selector(createInitialWorkbenchState().projects[0]!),
  useWorkbenchCommands: () => ({ generation: {}, widgets: {} }),
}));
vi.mock('./useInvocationState', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useInvocationState: () => ({
    invocation: { sourceId: 'generate', destination: 'gallery', sourceLocked: false, destinationLocked: false },
    placedTypeIds: new Set(['generate']),
  }),
}));
vi.mock('./InvokeButton', () => ({ InvokeButton: () => <Button>Invoke</Button> }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { InvocationCluster } from './InvocationCluster';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  accountLifecycle.invalidate();
});

it('puts iterations after left-side Invoke with matching joins and preserves input focus when placement changes', async () => {
  await page.viewport(1024, 200);
  accountLifecycle.activate('invocation-order-test');
  await patchWorkbenchPreferences({ showInvokeButtonOnLeft: false });
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root.render(
      <ChakraProvider value={system}>
        <InvocationCluster />
      </ChakraProvider>
    );
  });
  const input = host.querySelector<HTMLInputElement>('input')!;
  const iterations = input.closest<HTMLElement>('[data-scope="number-input"][data-part="root"]')!;
  const invoke = [...host.querySelectorAll('button')].find((button) => button.textContent === 'Invoke')!;
  expect(iterations.getBoundingClientRect().left).toBeLessThan(invoke.getBoundingClientRect().left);
  expect(getComputedStyle(input).borderTopLeftRadius).not.toBe('0px');
  await act(() => input.focus());
  const originalValue = input.value;

  await act(() => patchWorkbenchPreferences({ showInvokeButtonOnLeft: true }));

  expect(invoke.getBoundingClientRect().left).toBeLessThan(iterations.getBoundingClientRect().left);
  expect(invoke.getBoundingClientRect().right - iterations.getBoundingClientRect().left).toBeCloseTo(1, 1);
  expect(getComputedStyle(input).borderTopLeftRadius).toBe('0px');
  expect(getComputedStyle(input).borderTopRightRadius).not.toBe('0px');
  expect(getComputedStyle(iterations, '::after').left).toMatch(/^-/);
  expect(document.activeElement).toBe(input);
  expect(input.value).toBe(originalValue);

  await act(() => patchWorkbenchPreferences({ showInvokeButtonOnLeft: false }));
  expect(iterations.getBoundingClientRect().left).toBeLessThan(invoke.getBoundingClientRect().left);
  expect(document.activeElement).toBe(input);
});
