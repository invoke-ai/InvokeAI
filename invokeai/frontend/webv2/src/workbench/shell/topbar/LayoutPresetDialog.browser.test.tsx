import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { graphWidgetSources } from '@workbench/graphWidgets';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'common.cancel': 'Cancel',
        'topbar.presets.defaultDestination': 'Default destination',
        'topbar.presets.defaultSource': 'Default source',
        'topbar.presets.icon': 'Icon',
        'topbar.presets.iconPicker': 'Layout preset icon',
        'topbar.presets.name': 'Preset name',
        'topbar.presets.noInvocationSource': 'This layout does not contain an invocation source.',
      })[key] ?? key,
  }),
}));

import { LayoutPresetDialog } from './LayoutPresetDialog';

const BUILT_IN_DEFAULT_ROUTE = { destination: 'gallery', sourceId: 'workflow' } as const;
const BUILT_IN_SOURCE_OPTIONS = graphWidgetSources.filter(
  (source) => source.sourceId === 'generate' || source.sourceId === 'workflow'
);

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('LayoutPresetDialog', () => {
  it('edits identity and routing for a built-in preset within its saved arrangement', async () => {
    const onClose = vi.fn();
    const onSubmit = vi.fn();
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <LayoutPresetDialog
            defaultRoute={BUILT_IN_DEFAULT_ROUTE}
            iconId="workflow"
            isOpen
            name="Automate"
            sourceOptions={BUILT_IN_SOURCE_OPTIONS}
            submitLabel="Save preset"
            title="Edit layout preset"
            onClose={onClose}
            onSubmit={onSubmit}
          />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const nameInput = document.querySelector<HTMLInputElement>('input[name="layout-preset-name"]');
    expect(nameInput).not.toBeNull();
    expect(document.querySelector('[role="radiogroup"][aria-label="Layout preset icon"]')).not.toBeNull();
    expect(document.body.textContent).toContain('Default source');
    expect(document.body.textContent).toContain('Default destination');

    await act(() => userEvent.clear(nameInput!));
    await act(() => userEvent.type(nameInput!, 'Build'));
    const starIcon = document.querySelector<HTMLButtonElement>('button[aria-label="Star"]');
    expect(starIcon).not.toBeNull();
    await act(() => userEvent.click(starIcon!));

    const sourceSelect = document.querySelector<HTMLSelectElement>('select');
    expect(Array.from(sourceSelect?.options ?? []).map((option) => option.value)).toEqual(['generate', 'workflow']);
    expect(sourceSelect?.value).toBe('workflow');

    const destinationGroup = document.querySelector<HTMLElement>('[data-scope="segment-group"][data-part="root"]');
    expect(destinationGroup?.getAttribute('aria-label')).toBe('Default destination');

    const sourceTrigger = document.querySelector<HTMLButtonElement>('[data-scope="select"][data-part="trigger"]');
    expect(sourceTrigger).toBeDefined();
    expect(sourceTrigger?.getAttribute('aria-label')).toBe('Default source');
    await act(() => userEvent.click(sourceTrigger!));
    const generateOption = Array.from(
      document.querySelectorAll<HTMLElement>('[data-scope="select"][data-part="item"]')
    ).find((option) => option.textContent === 'Generate');
    expect(generateOption).toBeDefined();
    await act(() => userEvent.click(generateOption!));

    const canvasDestination = document.querySelector<HTMLInputElement>('input[type="radio"][value="canvas"]');
    expect(canvasDestination).toBeDefined();
    const canvasDestinationLabel = document.querySelector<HTMLLabelElement>(`label[for="${canvasDestination?.id}"]`);
    expect(canvasDestinationLabel).toBeDefined();
    await act(() => userEvent.click(canvasDestinationLabel!));

    const submit = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find(
      (button) => button.textContent === 'Save preset'
    );
    expect(submit).toBeDefined();
    await act(() => userEvent.click(submit!));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        defaultRoute: { destination: 'canvas', sourceId: 'generate' },
        iconId: 'star',
        name: 'Build',
      })
    );
    expect(onClose).toHaveBeenCalledOnce();
  });

  it('makes the fallback icon selected and keyboard-focusable for a retired icon id', async () => {
    host = document.createElement('div');
    document.body.append(host);
    root = createRoot(host);

    await act(async () => {
      root?.render(
        <ChakraProvider value={system}>
          <LayoutPresetDialog
            defaultRoute={BUILT_IN_DEFAULT_ROUTE}
            iconId="retired-icon"
            isOpen
            name="Legacy preset"
            sourceOptions={BUILT_IN_SOURCE_OPTIONS}
            submitLabel="Save preset"
            title="Edit layout preset"
            onClose={vi.fn()}
            onSubmit={vi.fn()}
          />
        </ChakraProvider>
      );
      await new Promise<void>((resolve) => {
        globalThis.setTimeout(resolve, 50);
      });
    });

    const fallback = document.querySelector<HTMLButtonElement>('[role="radio"][data-icon-id="layout-grid"]');
    expect(fallback).toHaveAttribute('aria-checked', 'true');
    expect(fallback?.tabIndex).toBe(0);
    fallback?.focus();
    await act(() => userEvent.keyboard('{ArrowRight}'));

    const next = document.querySelector<HTMLButtonElement>('[role="radio"][data-icon-id="layout-dashboard"]');
    expect(next).toHaveAttribute('aria-checked', 'true');
    expect(next).toHaveFocus();
  });
});
