import type { CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { RegionalGuidanceSettings } from './RegionalGuidanceSettings';

let selectedBase: string | null = null;
vi.mock('./useSelectedModelBase', () => ({ useSelectedModelBase: () => selectedBase }));

type Layer = ComponentProps<typeof RegionalGuidanceSettings>['layer'];

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const createLayer = (overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}): Layer =>
  ({
    autoNegative: false,
    id: 'rg-1',
    isEnabled: true,
    isLocked: false,
    mask: { bitmap: null, fill: { color: '#ff0000', style: 'solid' } },
    name: 'Region 1',
    negativePrompt: null,
    opacity: 1,
    positivePrompt: 'a cat',
    referenceImages: [],
    type: 'regional_guidance',
    ...overrides,
  }) as unknown as Layer;

const render = async (base: string | null, layer: Layer) => {
  selectedBase = base;
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <RegionalGuidanceSettings engine={null} layer={layer} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

afterEach(async () => {
  await act(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  resetArchitectureCapabilities();
});

const alerts = () => [...host!.querySelectorAll('[role="alert"]')].map((el) => el.textContent);
const hasNegativeField = () =>
  host!.querySelector('[aria-label="widgets.layers.regionalGuidance.negativePrompt"]') !== null;
const hasAutoNegativeSwitch = () =>
  Boolean(host!.textContent?.includes('widgets.layers.regionalGuidance.autoNegative'));

describe('RegionalGuidanceSettings per model base', () => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  it('offers both prompt polarities on an SD base', async () => {
    await render('sd-1', createLayer());
    expect(alerts()).toEqual([]);
    expect(hasNegativeField()).toBe(true);
    expect(hasAutoNegativeSwitch()).toBe(true);
  });

  it('hides the negative controls on a positive-only base even when the layer still holds a negative', async () => {
    await render('anima', createLayer({ autoNegative: true, negativePrompt: 'blurry' }));
    expect(alerts()).toEqual([]);
    expect(hasNegativeField()).toBe(false);
    expect(hasAutoNegativeSwitch()).toBe(false);
  });

  it('warns that a base with no regional path skips the layer', async () => {
    await render('sd-3', createLayer());
    expect(alerts()).toEqual(['widgets.layers.regionalGuidance.unsupportedModel']);
    expect(hasNegativeField()).toBe(true);
  });

  it('offers everything while no model is selected', async () => {
    await render(null, createLayer());
    expect(alerts()).toEqual([]);
    expect(hasNegativeField()).toBe(true);
  });
});

describe('RegionalGuidanceSettings before the capability table arrives', () => {
  it('does not accuse a supported model of lacking a regional path', async () => {
    // Missing capabilities must not announce supported regional models as unsupported while rendering their
    // controls.
    await render('sd-1', createLayer());

    expect(alerts()).toEqual([]);
    expect(hasNegativeField()).toBe(true);
    expect(hasAutoNegativeSwitch()).toBe(true);
  });

  it('renders the real answer once the table lands, without remounting', async () => {
    await render('sd-3', createLayer());
    expect(alerts()).toEqual([]);

    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    expect(alerts()).toEqual(['widgets.layers.regionalGuidance.unsupportedModel']);
  });

  it('hides the negative controls once the table says the base ignores them', async () => {
    await render('anima', createLayer({ autoNegative: true, negativePrompt: 'blurry' }));
    expect(hasNegativeField()).toBe(true);

    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    expect(hasNegativeField()).toBe(false);
    expect(hasAutoNegativeSwitch()).toBe(false);
  });
});
