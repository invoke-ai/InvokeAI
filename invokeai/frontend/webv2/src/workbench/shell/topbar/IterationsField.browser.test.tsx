import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { Group } from '@platform/ui/Group';
import { system } from '@theme/system';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';

const harness = vi.hoisted(() => ({ patchValues: vi.fn(), project: null as unknown }));

vi.mock('@workbench/WorkbenchContext', async (importOriginal) => ({
  ...(await importOriginal<Record<string, unknown>>()),
  useActiveProjectSelector: (selector: (project: Project) => unknown) => selector(harness.project as Project),
  useWorkbenchCommands: () => ({
    generation: { setBatchCount: vi.fn() },
    widgets: { patchValues: harness.patchValues },
  }),
}));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));

import { createInitialWorkbenchState, workbenchReducer } from '@workbench/workbenchState.testing';

import { IterationsField } from './IterationsField';

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

afterEach(async () => {
  harness.patchValues.mockReset();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

const renderForUpscale = async (mode: 'refine' | 'model-only', withInvoke = false) => {
  let state = workbenchReducer(createInitialWorkbenchState(), {
    type: 'patchWidgetValues',
    values: { batchCount: 3, mode },
    widgetId: 'upscale',
  });
  state = workbenchReducer(state, { sourceId: 'upscale', type: 'setInvocationSource' });
  harness.project = state.projects.find((project) => project.id === state.activeProjectId);
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);

  await act(() => {
    root?.render(
      <ChakraProvider value={system}>
        {withInvoke ? (
          // The cluster's join: Invoke follows the field in an attached group.
          <Group attached>
            <IterationsField />
            <Button size="lg" zIndex="2">
              Invoke
            </Button>
          </Group>
        ) : (
          <IterationsField />
        )}
      </ChakraProvider>
    );
  });

  return host.querySelector<HTMLInputElement>('input[aria-label="topbar.iterations.label"]')!;
};

const stepUp = async (input: HTMLInputElement) => {
  await act(() => {
    input.focus();
    input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'ArrowUp' }));
  });
};

describe('IterationsField for Upscale', () => {
  it("edits Refine's iterations", async () => {
    const input = await renderForUpscale('refine');

    expect(input.value).toBe('3');
    expect(input.getAttribute('aria-describedby')).toBeNull();

    await stepUp(input);

    expect(harness.patchValues).toHaveBeenCalledWith('upscale', { batchCount: 4 });
  });

  it('holds Model only at one run and says why, keeping the field focusable', async () => {
    const input = await renderForUpscale('model-only');

    expect(input.value).toBe('1');
    expect(input.disabled).toBe(false);
    expect(input.getAttribute('aria-disabled')).toBe('true');
    expect(document.getElementById(input.getAttribute('aria-describedby') ?? '')?.textContent).toBe(
      'topbar.iterations.fixedModelOnly'
    );

    await stepUp(input);

    expect(document.activeElement).toBe(input);
    expect(input.value).toBe('1');
    expect(harness.patchValues).not.toHaveBeenCalled();
  });
});

describe('IterationsField beside Invoke', () => {
  it("runs its frame under Invoke's rounded corners while its steppers stay clear of Invoke", async () => {
    const input = await renderForUpscale('refine', true);
    const field = input.parentElement!;
    const invoke = host!.querySelector('button:not([data-part])')!.getBoundingClientRect();
    const steppers = field.querySelector('[data-part="control"]')!.getBoundingClientRect();
    // Invoke keeps full control corners in the cluster; the field's own leading corner shows that radius.
    const cornerRadius = Number.parseFloat(getComputedStyle(field).borderTopLeftRadius);

    expect(cornerRadius).toBeGreaterThan(0);
    // The top and bottom borders reach where Invoke's edges turn straight, so no notch opens beside its corners.
    expect(input.getBoundingClientRect().right).toBeGreaterThanOrEqual(invoke.left + cornerRadius);
    expect(steppers.right).toBeLessThanOrEqual(invoke.left);
  });
});
