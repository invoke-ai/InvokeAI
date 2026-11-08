/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { ProjectSummary } from '@workbench/projects/library';

import { ChakraProvider } from '@chakra-ui/react';
import { isModalPresent } from '@platform/ui/modalPresence';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

/** Holds the dialog's module back, as a slow connection does on the first delete. */
const chunk = vi.hoisted(() => {
  let arrive = () => {};
  const arrived = new Promise<void>((resolve) => {
    arrive = resolve;
  });

  return { arrive, arrived };
});

vi.mock('@workbench/projects/components/DeleteProjectDialog', async () => {
  await chunk.arrived;

  return {
    DeleteProjectDialog: ({ isOpen }: { isOpen: boolean }) => <div data-open={String(isOpen)} data-testid="delete" />,
  };
});
vi.mock('./useProjectCardActions', () => ({
  useProjectCardActions: () => ({
    delete: () => Promise.resolve(),
    duplicate: () => {},
    export: () => {},
    rename: () => Promise.resolve(),
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
vi.mock('@tanstack/react-router', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useNavigate: () => () => {},
  Link: ({ children, ...props }: { children?: unknown } & Record<string, unknown>) => (
    <a href="/app" {...props}>
      {children as never}
    </a>
  ),
}));

const { ProjectActionsMenuProvider, useProjectActionsMenuTrigger } = await import('./ProjectActionsMenuHost');

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const summary = {
  id: 'one',
  minimumCanvasSchemaVersion: 3,
  name: 'Project one',
  schemaVersion: 1,
  updatedAt: 0,
} as unknown as ProjectSummary;
const menuTarget = { isPinned: false, onTogglePin: () => {}, summary };

const Card = () => {
  const trigger = useProjectActionsMenuTrigger(menuTarget);

  return (
    <button type="button" onClick={trigger.onClick} onPointerDown={trigger.onPointerDown}>
      actions
    </button>
  );
};

const host = document.createElement('div');
const root = createRoot(host);

afterEach(async () => {
  await act(() => root.unmount());
  host.remove();
});

it('suspends shortcuts from the request that opens the delete dialog, while its module is still loading', async () => {
  document.body.append(host);
  await act(() =>
    root.render(
      <ChakraProvider value={system}>
        <ProjectActionsMenuProvider>
          <Card />
        </ProjectActionsMenuProvider>
      </ChakraProvider>
    )
  );

  await act(() => userEvent.click(host.querySelector('button')!));
  await act(() => userEvent.click(document.querySelector<HTMLElement>('[role="menuitem"][data-value="delete"]')!));
  expect(document.querySelector('[data-testid="delete"]')).toBeNull();
  expect(isModalPresent()).toBe(true);

  await act(async () => {
    chunk.arrive();
    await chunk.arrived;
  });
  await expect.poll(() => document.querySelector('[data-testid="delete"]')?.getAttribute('data-open')).toBe('true');
  // The stand-in steps aside for the loaded dialog, which this test replaces with one that announces nothing.
  expect(isModalPresent()).toBe(false);
});
