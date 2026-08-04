/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { RebalancePreset } from '@features/generation/core/conditioningRebalance';
import type { GenerateSettings } from '@features/generation/core/types';

import { ChakraProvider } from '@chakra-ui/react';
import {
  DEFAULT_KREA2_REBALANCE_MULTIPLIER,
  DEFAULT_KREA2_REBALANCE_WEIGHTS,
  NEUTRAL_KREA2_REBALANCE_WEIGHTS,
} from '@features/generation/core/conditioningRebalance';
import { system } from '@theme/system';
import i18next from 'i18next';
import { act, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { initReactI18next, I18nextProvider } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GenerateConditioningRebalanceField } from './GenerateConditioningRebalanceField';
import { GenerationUiProvider, type GenerationUiAdapter } from './GenerationUiContext';

const i18n = i18next.createInstance();
await i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          generate: {
            krea2Rebalance: 'Conditioning rebalance',
            krea2RebalanceAxisDeep: 'deep',
            krea2RebalanceAxisShallow: 'shallow',
            krea2RebalanceGain: 'Gain',
            krea2RebalanceGainReset: 'Reset gain',
            krea2RebalanceHelp: 'Scales conditioning toward the prompt before denoising.',
            krea2RebalancePreset: 'Rebalance preset',
            krea2RebalancePresetActions: 'Preset actions',
            krea2RebalancePresetCustom: 'Custom',
            krea2RebalancePresetDelete: 'Delete preset',
            krea2RebalancePresetDeleteBody: 'Delete "{{name}}"?',
            krea2RebalancePresetName: 'Preset name',
            krea2RebalancePresetRename: 'Rename preset',
            krea2RebalancePresetSave: 'Save current as preset',
            krea2RebalancePresetSaveAction: 'Save',
            krea2RebalanceReset: 'Reset to default curve',
            krea2RebalanceTapLabel: 'Tap {{tap}}, encoder layer {{layer}}',
            krea2RebalanceTapReadout: 'tap {{tap}} · {{value}}',
            krea2RebalanceWeights: 'Per-tap weights',
            krea2RebalanceWeightsInvalid: 'Enter exactly {{count}} comma-separated numbers.',
          },
        },
      },
    },
  },
});

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const onCommit = vi.fn();
const onCommitImmediate = vi.fn();
const savePreset = vi.fn();
const renamePreset = vi.fn();
const removePreset = vi.fn();

let customPresets: RebalancePreset[] = [];

/**
 * The field reads exactly one sub-port and three settings keys. Building the whole
 * adapter would only couple this test to parts of the port it never touches.
 */
const buildAdapter = (): GenerationUiAdapter =>
  ({
    rebalancePresets: {
      presets: customPresets,
      remove: removePreset,
      rename: renamePreset,
      save: savePreset,
    },
  }) as unknown as GenerationUiAdapter;

const buildSettings = (overrides: Partial<GenerateSettings> = {}): GenerateSettings =>
  ({
    krea2RebalanceEnabled: true,
    krea2RebalanceMultiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
    krea2RebalanceWeights: DEFAULT_KREA2_REBALANCE_WEIGHTS,
    ...overrides,
  }) as unknown as GenerateSettings;

/** Mirrors the real form: commits land back on the rendered settings. */
const Harness = ({ initial }: { initial: Partial<GenerateSettings> }) => {
  const [settings, setSettings] = useState(() => buildSettings(initial));
  const apply = (patch: Partial<GenerateSettings>) => setSettings((current) => ({ ...current, ...patch }));

  return (
    <GenerateConditioningRebalanceField
      settings={settings}
      onCommit={(patch) => {
        onCommit(patch);
        apply(patch);
      }}
      onCommitImmediate={(patch) => {
        onCommitImmediate(patch);
        apply(patch);
      }}
    />
  );
};

const renderField = async (initial: Partial<GenerateSettings> = {}) => {
  await act(() =>
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <GenerationUiProvider adapter={buildAdapter()}>
            <Harness initial={initial} />
          </GenerationUiProvider>
        </ChakraProvider>
      </I18nextProvider>
    )
  );
};

const interact = async (run: () => void) => {
  await act(async () => {
    run();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 40);
    });
  });
};

/** Scoped by orientation: the Gain slider's thumb is a `role="slider"` too. */
const bars = () => [...document.querySelectorAll('[role="slider"][aria-orientation="vertical"]')];

const byLabel = (label: string): HTMLElement | null =>
  document.querySelector(`[aria-label="${label}"]`) as HTMLElement | null;

const weightsInput = (): HTMLInputElement => {
  const input = byLabel('Per-tap weights');

  if (!(input instanceof HTMLInputElement)) {
    throw new Error('weights input did not render');
  }

  return input;
};

/** React tracks its own value on the node, so a plain assignment is ignored. */
const typeInto = (input: HTMLInputElement, value: string) => {
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;

  setter?.call(input, value);
  input.dispatchEvent(new Event('input', { bubbles: true }));
};

beforeEach(() => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  customPresets = [];
  onCommit.mockClear();
  onCommitImmediate.mockClear();
  savePreset.mockClear();
  renamePreset.mockClear();
  removePreset.mockClear();
});

afterEach(async () => {
  await act(() => root?.unmount());
  document
    .querySelectorAll('[data-scope="select"][data-part="positioner"], [data-scope="menu"][data-part="positioner"]')
    .forEach((element) => element.remove());
  host?.remove();
  host = null;
  root = null;
});

describe('GenerateConditioningRebalanceField', () => {
  it('shows only the switch and a preview while disabled', async () => {
    await renderField({ krea2RebalanceEnabled: false });

    expect(document.body.textContent).toContain('Conditioning rebalance');
    expect(bars()).toHaveLength(0);
    expect(byLabel('Per-tap weights')).toBeNull();
    // The collapsed row still previews the curve.
    expect(document.querySelector('svg[aria-hidden]')).not.toBeNull();
  });

  it('turns the editor on from the collapsed row', async () => {
    await renderField({ krea2RebalanceEnabled: false });

    const toggle = byLabel('Conditioning rebalance');

    await interact(() => toggle?.click());

    expect(onCommitImmediate).toHaveBeenCalledWith({ krea2RebalanceEnabled: true });
    expect(bars()).toHaveLength(12);
  });

  it('labels the axis by encoder depth rather than a frequency metaphor', async () => {
    await renderField();

    expect(document.body.textContent).toContain('shallow');
    expect(document.body.textContent).toContain('deep');
    expect(bars()[0]?.getAttribute('aria-label')).toBe('Tap 1, encoder layer 2');
  });

  it('disables reset at the backend defaults and restores them once edited', async () => {
    await renderField();

    const reset = byLabel('Reset to default curve');

    expect(reset?.getAttribute('disabled')).not.toBeNull();

    await interact(() => typeInto(weightsInput(), NEUTRAL_KREA2_REBALANCE_WEIGHTS));

    expect(byLabel('Reset to default curve')?.getAttribute('disabled')).toBeNull();

    await interact(() => byLabel('Reset to default curve')?.click());

    expect(onCommitImmediate).toHaveBeenLastCalledWith({
      krea2RebalanceMultiplier: DEFAULT_KREA2_REBALANCE_MULTIPLIER,
      krea2RebalanceWeights: DEFAULT_KREA2_REBALANCE_WEIGHTS,
    });
    expect(weightsInput().value).toBe(DEFAULT_KREA2_REBALANCE_WEIGHTS);
  });

  it('commits a typed vector the backend would accept', async () => {
    await renderField();

    await interact(() => typeInto(weightsInput(), NEUTRAL_KREA2_REBALANCE_WEIGHTS));

    expect(onCommitImmediate).toHaveBeenLastCalledWith({ krea2RebalanceWeights: NEUTRAL_KREA2_REBALANCE_WEIGHTS });
    expect(bars()[0]?.getAttribute('aria-valuenow')).toBe('1');
  });

  it('reports an invalid vector without clobbering the bars', async () => {
    await renderField();

    await interact(() => typeInto(weightsInput(), '1,2,three'));

    expect(document.body.textContent).toContain('Enter exactly 12 comma-separated numbers.');
    // The last valid vector still drives the editor.
    expect(bars()[8]?.getAttribute('aria-valuenow')).toBe('5');
    expect(onCommitImmediate).not.toHaveBeenCalled();
  });

  it('holds a half-typed vector instead of normalizing it under the cursor', async () => {
    await renderField();

    await interact(() => typeInto(weightsInput(), '1.0,1.0,'));

    expect(weightsInput().value).toBe('1.0,1.0,');
  });

  it('falls back to Custom once the values leave every preset', async () => {
    await renderField();

    expect(document.body.textContent).toContain('Default');

    await interact(() => typeInto(weightsInput(), '1,1,1,1,1,1,1,2.5,5,1.1,4,7'));

    expect(document.body.textContent).toContain('Custom');
  });

  it('saves the current curve as a named preset', async () => {
    await renderField();

    await interact(() => byLabel('Preset actions')?.click());

    const saveItem = [...document.querySelectorAll('[role="menuitem"]')].find((item) =>
      item.textContent?.includes('Save current as preset')
    );

    await interact(() => (saveItem as HTMLElement | undefined)?.click());

    const nameInput = document.querySelector('[role="dialog"] input');

    if (!(nameInput instanceof HTMLInputElement)) {
      throw new Error('preset name dialog did not open');
    }

    await interact(() => typeInto(nameInput, 'Punchy'));
    await interact(() => nameInput.form?.requestSubmit());

    expect(savePreset).toHaveBeenCalledWith('Punchy', DEFAULT_KREA2_REBALANCE_WEIGHTS, 4);
  });
});
