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

vi.mock('@workbench/settings/useLayeredSetting', () => ({
  useLayeredSetting: () => ({
    disabled: false,
    layer: undefined,
    set: (clusterEps: number | null) => settings.patch({ clusterEps }),
    value: useSyncExternalStore(settings.subscribe, settings.get, settings.get),
  }),
}));

const { ClusterStrengthField } = await import('./ClusterStrengthField');

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const SETTLE_MS = 900;

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

const setValue = async (value: string) => {
  const input = spinner();

  await act(() => {
    input.focus();
    setNativeValue.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
};

const field = imageMapSettingsContribution.fields.find((entry) => entry.id === 'clusterEps')!;

const render = async (resolved: number | null) => {
  imageMapStore.setSnapshot({
    clusterLabels: null,
    clusterLabelsEps: null,
    clusterLabelsHash: null,
    data: {
      clusterEps: resolved,
      modelId: null,
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

const autoBadgeShown = (): boolean => {
  const badge = host?.querySelector('[data-testid="cluster-strength-auto"]');

  return badge instanceof HTMLElement && globalThis.getComputedStyle(badge).visibility === 'visible';
};

const spinner = (): HTMLInputElement => {
  const input = host?.querySelector('input[type="number"]');

  if (!(input instanceof HTMLInputElement)) {
    throw new TypeError('the clustering-strength spinner is not in the document');
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
    expect(spinner().value).toBe('0.0945');
    expect(autoBadgeShown()).toBe(true);
  });

  it('does not mistake a half-typed number for a cleared box', async () => {
    // Faked so the keys stay one burst however slowly a loaded browser delivers them: on the real clock, a "." landing
    // past the debounce finds the box genuinely empty and commits that as a deliberate clear.
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    // `type="number"` reports an empty value for "0." too. Reading that as "use the heuristic" would throw the user's
    // setting away mid-keystroke.
    await render(0.0945);
    await setValue('0.25');
    await settleFake();
    settings.patch.mockClear();

    await userEvent.type(spinner(), '{selectall}{Backspace}.', { skipClick: true });
    expect(spinner().value).toBe('');
    expect(spinner().validity.badInput).toBe(true);

    await settleFake();

    expect(settings.patch).not.toHaveBeenCalled();
  });

  it('does not complain on the way through an incomplete number', async () => {
    await render(0.0945);
    await setValue('');
    await userEvent.type(spinner(), '0.15');

    expect(host?.textContent).not.toContain('clusterStrengthRange');

    await settle();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: 0.15 });
  });

  it('refills the box when cleared from the heuristic it was already on', async () => {
    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    await render(0.0945);

    await setValue('');
    await settleFake();

    expect(settings.patch).toHaveBeenCalledWith({ clusterEps: null });
    expect(spinner().value).toBe('0.0945');
    expect(autoBadgeShown()).toBe(true);
  });

  it('does not leave the refilled heuristic for the next keystroke to append to', async () => {
    // Clearing refills the box with the derived value — while the caret is still in it. Without selecting the refilled
    // text, typing "0.5" next lands "0.09450.5" and the map reclusters at a number nobody chose.
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
