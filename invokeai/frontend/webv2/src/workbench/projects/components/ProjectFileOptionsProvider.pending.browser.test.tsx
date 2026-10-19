/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import { accountLifecycle, captureAccountScope } from '@platform/state/accountLifecycle';
import { isModalPresent } from '@platform/ui/modalPresence';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';

/** Holds the dialog's module back, as a slow connection does on the first export. */
const chunk = vi.hoisted(() => {
  let arrive = () => {};
  const arrived = new Promise<void>((resolve) => {
    arrive = resolve;
  });

  return { arrive, arrived };
});

vi.mock('./ProjectFileOptionsDialog', async () => {
  await chunk.arrived;

  return {
    ProjectFileOptionsDialog: ({ isOpen }: { isOpen: boolean }) => (
      <div data-open={String(isOpen)} data-testid="options" />
    ),
  };
});

import { ProjectFileOptionsProvider, useProjectFileOptions } from './ProjectFileOptionsProvider';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const Harness = () => {
  const { requestExportOptions } = useProjectFileOptions();

  return (
    <button type="button" onClick={() => void requestExportOptions('Typography', captureAccountScope())}>
      Export
    </button>
  );
};

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
  accountLifecycle.invalidate();
});

it('suspends shortcuts from the request that opens the dialog, while its module is still loading', async () => {
  accountLifecycle.activate('options-pending-test');
  document.body.append(host);
  await act(() =>
    root.render(
      <ProjectFileOptionsProvider>
        <Harness />
      </ProjectFileOptionsProvider>
    )
  );

  await act(() => host.querySelector('button')!.click());
  expect(host.querySelector('[data-testid="options"]')).toBeNull();
  expect(isModalPresent()).toBe(true);

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });
  await expect.poll(() => host.querySelector('[data-testid="options"]')).not.toBeNull();
  // The stand-in steps aside for the loaded dialog, which this test replaces with one that announces nothing.
  expect(isModalPresent()).toBe(false);
});
