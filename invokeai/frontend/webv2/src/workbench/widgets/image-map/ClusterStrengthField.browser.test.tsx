import type { SettingFieldProps } from '@platform/ui/settings/contracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { imageMapStore } from '@workbench/image-map/imageMapStore';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { imageMapSettingsContribution } from './settingsContribution';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string) =>
      ({
        'settingsDialog.fields.clusterStrength': 'Clustering strength',
        'settingsDialog.fields.clusterStrengthAuto': 'auto',
        'settingsDialog.fields.clusterStrengthHint': 'Higher values create larger clusters',
      })[key] ?? key,
  }),
}));

// A real subscribable stand-in for the widget's persisted values: committing
// has to re-render the field, which is how the box learns the edit landed.
const settings = vi.hoisted(() => {
  const listeners = new Set<() => void>();
  let value: number | null = null;

  return {
    get: () => value,
    patch: vi.fn((values: Record<string, unknown>) => {
      value = (values.clusterEps as number | null) ?? null;
      for (const listener of listeners) {
        listener();
      }
    }),
    reset: () => {
      value = null;
      listeners.clear();
    },
    subscribe: (listener: () => void) => {
      listeners.add(listener);

      return () => listeners.delete(listener);
    },
  };
});

vi.mock('@workbench/settings/useWidgetSettingsTarget', () => ({
  useWidgetSettingsTarget: () => ({
    disabled: false,
    patch: settings.patch,
    value: useSyncExternalStore(settings.subscribe, settings.get, settings.get),
  }),
}));

const { ClusterStrengthField } = await import('./ClusterStrengthField');

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** Longer than the field's own debounce, so a commit has certainly fired. */
const SETTLE_MS = 900;

/**
 * Real-time settle, for the tests that type through the real keyboard. Everything else fakes the clock (see
 * `setValue`): this file runs beside other browser test files that share the page's focus, and seconds of real
 * typing and waiting here were enough to knock their popovers and menus closed on CI.
 */
const settle = () =>
  act(async () => {
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, SETTLE_MS);
    });
  });

const settleFake = () =>
  act(() => {
    vi.advanceTimersByTime(SETTLE_MS);
  });

const setNativeValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')!.set!;

/**
 * A whole-value edit (paste, fill, or clear) without driving the page's keyboard: the native setter plus an `input`
 * event is what React's onChange observes. Keystroke-level behaviour — a `type="number"` box's `badInput`, partial
 * numbers, typing after a refill — still goes through `userEvent`, since only real keys produce it.
 */
const setValue = async (value: string) => {
  const input = spinner();

  await act(() => {
    input.focus();
    setNativeValue.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

const field = imageMapSettingsContribution.fields.find((entry) => entry.id === 'clusterEps')!;

/** `resolved` is what the server says it clustered with. */
const render = async (resolved: number | null) => {
  imageMapStore.setSnapshot({
    clusterLabels: null,
    clusterLabelsEps: null,
    clusterLabelsHash: null,
    data: {
      clusterEps: resolved,
      modelName: null,
      pointCount: 1,
      points: [],
      stale: false,
      state: 'ready',
      updatedAt: null,
      visibleHash: null,
    },
    error: null,
    indexCounts: null,
    indexUpdatedAt: null,
    loadState: 'loaded',
    renderError: null,
  });

  await act(() =>
    root?.render(
      <ChakraProvider value={system}>
        <ClusterStrengthField field={field} surface="dialog" {...({} as Partial<SettingFieldProps>)} />
      </ChakraProvider>
    )
  );
};

/** The "auto" badge is hidden rather than unmounted, to keep the row from shifting. */
const autoBadgeShown = (): boolean => {
  const badge = host?.querySelector('[data-testid="cluster-strength-auto"]');

  return badge instanceof HTMLElement && globalThis.getComputedStyle(badge).visibility === 'visible';
};

const spinner = (): HTMLInputElement => {
  const input = host?.querySelector('input[type="number"]');

  if (!(input instanceof HTMLInputElement)) {
    throw new Error('the clustering-strength spinner is not in the document');
  }

  return input;
};

beforeEach(() => {
  settings.reset();
  settings.patch.mockClear();
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
});

afterEach(async () => {
  vi.useRealTimers();
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
});

describe('ClusterStrengthField', () => {
  it('shows the strength the server derived, marked as not the user’s', async () => {
    await render(0.0945);

    expect(spinner().value).toBe('0.0945');
    expect(autoBadgeShown()).toBe(true);
    expect(host?.textContent).toContain('Higher values create larger clusters');
  });

  it('commits a typed strength once, after the edit settles', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);

    await setValue('0.25');
    expect(settings.patch).not.toHaveBeenCalled();

    await settleFake();

    expect(settings.patch).toHaveBeenCalledTimes(1);
    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.25 });
    expect(autoBadgeShown()).toBe(false);
  });

  it('hands the choice back to the heuristic when cleared', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);
    await setValue('0.25');
    await settleFake();
    settings.patch.mockClear();

    await setValue('');
    await settleFake();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: null });
    // And the box refills with the derived value rather than staying empty.
    expect(spinner().value).toBe('0.0945');
    expect(autoBadgeShown()).toBe(true);
  });

  it('does not mistake a half-typed number for a cleared box', async () => {
    // `type="number"` reports an empty value for "0." too. Reading that as
    // "use the heuristic" would throw the user's setting away mid-keystroke.
    await render(0.0945);
    await setValue('0.25');
    await settle();
    settings.patch.mockClear();

    // One keyboard sequence: clear() waits long enough to commit the intentionally empty box before type() runs.
    await userEvent.type(spinner(), '{selectall}{Backspace}.', { skipClick: true });
    // The box reads as empty here, which is the whole trap.
    expect(spinner().value).toBe('');
    expect(spinner().validity.badInput).toBe(true);

    await settle();

    expect(settings.patch).not.toHaveBeenCalled();
  });

  it('does not complain on the way through an incomplete number', async () => {
    // "0.15" passes through "0", which is below the minimum. Validating per
    // keystroke would flash a range error at a user typing a fine value.
    await render(0.0945);
    await setValue('');
    await userEvent.type(spinner(), '0.15');

    expect(host?.textContent).not.toContain('clusterStrengthRange');

    await settle();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.15 });
  });

  it('refills the box when cleared from the heuristic it was already on', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    // Nothing was chosen, so committing null changes no state and the sync
    // effect never fires. Without an explicit refill the box sits empty while
    // the map clusters at a strength the control is no longer showing.
    await render(0.0945);

    await setValue('');
    await settleFake();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: null });
    expect(spinner().value).toBe('0.0945');
    expect(autoBadgeShown()).toBe(true);
  });

  it('does not leave the refilled heuristic for the next keystroke to append to', async () => {
    // Clearing refills the box with the derived value — while the caret is
    // still in it. Without selecting the refilled text, typing "0.5" next
    // lands "0.09450.5" and the map reclusters at a number nobody chose.
    await render(0.0945);
    await setValue('0.25');
    await settle();
    settings.patch.mockClear();

    spinner().focus();
    await setValue('');
    await settle();
    expect(spinner().value).toBe('0.0945');

    await userEvent.type(spinner(), '0.5');
    await settle();

    expect(settings.patch).toHaveBeenLastCalledWith({ clusterEps: 0.5 });
    expect(spinner().value).toBe('0.5');
  });

  it('keeps every digit of a value with more precision than the box shows', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    // forDisplay trims to three significant digits. Rewriting the box with
    // the trimmed value would delete the keystroke that produced it.
    await render(0.0945);

    await setValue('0.1234');
    await settleFake();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.1234 });
    expect(spinner().value).toBe('0.1234');
  });

  it('commits an edit the dialog closes on top of', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);
    await setValue('0.25');

    // Unmounted inside the debounce window, which is what closing the
    // settings dialog does if the field never blurs.
    await act(() => root?.unmount());
    root = null;

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.25 });
  });

  it('commits an edit the user clicks away from', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);
    await setValue('0.25');
    await act(() => {
      spinner().blur();
    });

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.25 });
  });

  it('refuses a strength the endpoint would reject', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);

    await setValue('9');
    await settleFake();

    expect(settings.patch).not.toHaveBeenCalled();
    expect(host?.textContent).toContain('clusterStrengthRange');
  });
});
