import type { CanvasRegionalGuidanceLayerContract } from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import {
  resetArchitectureCapabilities,
  setArchitectureCapabilities,
} from '@features/generation/core/architectureCapabilities';
import { architectureCapabilitiesFixture } from '@features/generation/core/architectureCapabilities.testing';
import { flushWorkbenchDrafts } from '@platform/react/draftRegistry';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentLayer } from '@workbench/canvas-engine/document/documentIndex';
import { createInstance } from 'i18next';
import { act, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { page, userEvent } from 'vitest/browser';

import { RegionalGuidanceSettings } from './RegionalGuidanceSettings';

let selectedBase: string | null = null;
let selectedCfgScale: number | null = 7;
vi.mock('./useSelectedModelBase', () => ({
  useSelectedCfgScale: () => selectedCfgScale,
  useSelectedModelBase: () => selectedBase,
}));

type Layer = ComponentProps<typeof RegionalGuidanceSettings>['layer'];

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });
const english = createInstance();
await english.init({
  interpolation: { escapeValue: false },
  lng: 'en',
  resources: { en: { translation: await fetch('/locales/en.json').then((response) => response.json()) } },
});

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

const render = async (base: string | null, layer: Layer, translations = i18n) => {
  selectedBase = base;
  applyThemeToRoot('classic');
  if (!host) {
    host = document.createElement('div');
    host.style.width = '260px';
    document.body.append(host);
    root = createRoot(host);
  }
  await act(() => {
    root?.render(
      <I18nextProvider i18n={translations}>
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
  selectedCfgScale = 7;
  resetArchitectureCapabilities();
});

let stub: StructuralEngineStub;
const liveLayer = () => getDocumentLayer(stub.document(), 'rg-1') as CanvasRegionalGuidanceLayerContract;

const LiveSettings = () => {
  const layer = useSyncExternalStore(stub.subscribe, liveLayer);
  return <RegionalGuidanceSettings engine={stub.engine as never} layer={layer} />;
};

const renderLive = async (base: string, overrides: Partial<CanvasRegionalGuidanceLayerContract> = {}) => {
  stub = createStructuralEngineStub({
    layers: [layerContract('rg-1', 'regional_guidance', { positivePrompt: 'a shop front', ...overrides })],
  });
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
          <LiveSettings />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
};

const alerts = () => [...host!.querySelectorAll('[role="alert"]')].map((el) => el.textContent);
const hasNegativeField = () =>
  host!.querySelector('[aria-label="widgets.layers.regionalGuidance.negativePrompt"]') !== null;
const hasAutoNegativeSwitch = () =>
  Boolean(host!.textContent?.includes('widgets.layers.regionalGuidance.autoNegative'));
const negativeInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.negativePrompt' });
const addNegativeButton = () => page.getByRole('button', { name: 'widgets.layers.regionalGuidance.addNegativePrompt' });
const removeNegativeButton = () =>
  page.getByRole('button', { name: 'widgets.layers.regionalGuidance.removeNegativePrompt' });
const hasNegativeControls = () => hasNegativeField() || addNegativeButton().query() !== null;

describe('RegionalGuidanceSettings per model base', () => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  it.each([
    ['sdxl', true],
    ['flux', false],
  ])('colours and weights regional attention syntax on %s: %s', async (base, readsAttention) => {
    await render(base, createLayer({ positivePrompt: 'a cat+' }));
    const prompt = host!.querySelector('[aria-label="widgets.layers.regionalGuidance.positivePrompt"]')!;

    expect(host!.querySelector('pre') !== null).toBe(readsAttention);
    expect(prompt.matches('[data-prompt-attention-target="true"]')).toBe(readsAttention);
  });

  it('offers both prompt polarities on an SD base, the negative behind its add button until it has one', async () => {
    await render('sd-1', createLayer());
    expect(alerts()).toEqual([]);
    expect(hasNegativeField()).toBe(false);
    await expect.element(addNegativeButton()).toBeDisabled();
    expect(hasAutoNegativeSwitch()).toBe(true);

    await render('sd-1', createLayer({ negativePrompt: 'blurry' }));
    await expect.element(negativeInput()).toHaveValue('blurry');
    expect(addNegativeButton().query()).toBeNull();
  });

  it('warns that a base with no regional path skips the layer', async () => {
    await render('sd-3', createLayer());
    expect(alerts()).toEqual(['widgets.layers.regionalGuidance.unsupportedModel']);
    expect(hasNegativeControls()).toBe(true);
  });
});

describe('RegionalGuidanceSettings before the capability table arrives', () => {
  it('renders the real answer once the table lands, without remounting', async () => {
    await render('sd-3', createLayer());
    expect(alerts()).toEqual([]);

    await act(() => {
      setArchitectureCapabilities(architectureCapabilitiesFixture);
    });

    expect(alerts()).toEqual(['widgets.layers.regionalGuidance.unsupportedModel']);
  });
});

describe('RegionalGuidanceSettings on a layout base', () => {
  const kindGroup = () => page.getByRole('radiogroup', { name: 'widgets.layers.regionalGuidance.layoutKind' });
  const pickKind = (kind: 'object' | 'character' | 'text') =>
    act(() => userEvent.click(page.getByText(`widgets.layers.regionalGuidance.layoutKinds.${kind}`, { exact: true })));
  const promptInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.positivePrompt' });
  const visibleTextInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.visibleText' });
  const announced = () => host!.querySelector('[role="status"]')?.textContent ?? null;
  const descriptionOf = (element: Element) =>
    (element.getAttribute('aria-describedby') ?? '')
      .split(' ')
      .filter(Boolean)
      .map((id) => document.getElementById(id)?.textContent);

  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  it('offers the region type instead of the negative controls, and no type on a mask base', async () => {
    await render('moss-image', createLayer({ autoNegative: true, negativePrompt: 'blurry' }));
    await expect.element(kindGroup()).toBeInTheDocument();
    expect(hasNegativeControls()).toBe(false);
    expect(hasAutoNegativeSwitch()).toBe(false);
    expect(alerts()).toEqual([]);
    await render('sd-1', createLayer());
    expect(kindGroup().query()).toBeNull();
    expect(host!.querySelector('[role="status"]')).toBeNull();
  });

  it('records a type switch and the visible text as one undo step each, keeping text across a switch', async () => {
    await renderLive('moss-image');
    expect(visibleTextInput().query()).toBeNull();

    await pickKind('text');
    expect(liveLayer().layout).toEqual({ kind: 'text' });
    expect(announced()).toBe('widgets.layers.regionalGuidance.layoutMissingText');
    expect(descriptionOf(visibleTextInput().element())).toEqual(['widgets.layers.regionalGuidance.layoutMissingText']);

    await act(() => visibleTextInput().fill('OPEN'));
    expect(stub.commits).toHaveLength(1);
    await act(() => userEvent.tab());
    expect(stub.commits).toHaveLength(2);
    expect(liveLayer().layout).toEqual({ kind: 'text', text: 'OPEN' });
    expect(announced()).toBe('');
    expect(descriptionOf(visibleTextInput().element())).toEqual([]);

    await pickKind('character');
    expect(liveLayer().layout).toEqual({ kind: 'character', text: 'OPEN' });
    await pickKind('text');
    await expect.element(visibleTextInput()).toHaveValue('OPEN');

    await act(() => stub.engine.history.undo());
    await act(() => stub.engine.history.undo());
    await act(() => stub.engine.history.undo());
    expect(liveLayer().layout).toEqual({ kind: 'text' });
    await expect.element(visibleTextInput()).toHaveValue('');
    await act(() => stub.engine.history.undo());
    expect(Object.hasOwn(liveLayer(), 'layout')).toBe(false);
    expect(visibleTextInput().query()).toBeNull();
  });

  it('commits typed visible text when Invoke flushes drafts before the field blurs', async () => {
    await renderLive('moss-image', { layout: { kind: 'text' } });

    await act(() => visibleTextInput().fill('SALE'));
    expect(stub.commits).toHaveLength(0);
    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().layout).toEqual({ kind: 'text', text: 'SALE' });
  });

  it('leaves an undone caption undone when drafts flush later, showing the restored value', async () => {
    await renderLive('moss-image');

    await act(() => promptInput().fill('a cat'));
    await act(() => userEvent.tab());
    expect(liveLayer().positivePrompt).toBe('a cat');
    await act(() => stub.engine.history.undo());
    await expect.element(promptInput()).toHaveValue('a shop front');

    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().positivePrompt).toBe('a shop front');
    expect(stub.engine.history.canRedo()).toBe(true);
  });
});

describe('RegionalGuidanceSettings negative prompt', () => {
  beforeEach(() => {
    setArchitectureCapabilities(architectureCapabilitiesFixture);
  });

  it('adds an empty negative prompt as one undo step, focuses it, and records what is typed', async () => {
    await renderLive('sd-1');
    expect(hasNegativeField()).toBe(false);

    await act(() => userEvent.click(addNegativeButton()));

    expect(liveLayer().negativePrompt).toBe('');
    expect(stub.history.entries().past).toEqual(['widgets.layers.regionalGuidance.addNegativePrompt']);
    await expect.element(negativeInput()).toHaveFocus();
    expect(addNegativeButton().query()).toBeNull();

    await act(() => userEvent.keyboard('blurry'));
    await act(() => userEvent.tab());
    expect(liveLayer().negativePrompt).toBe('blurry');

    await act(() => negativeInput().fill(''));
    await act(() => userEvent.tab());
    expect(liveLayer().negativePrompt).toBe('');
    await expect.element(negativeInput()).toBeInTheDocument();
  });

  it('removes the negative prompt from the layer, hands focus to the add button, and undoes back', async () => {
    await renderLive('sd-1', { negativePrompt: 'blurry' });
    await expect.element(negativeInput()).toHaveValue('blurry');

    await act(() => userEvent.click(removeNegativeButton()));

    expect(liveLayer().negativePrompt).toBeNull();
    expect(stub.history.entries().past).toEqual(['widgets.layers.regionalGuidance.removeNegativePrompt']);
    expect(hasNegativeField()).toBe(false);
    await expect.element(addNegativeButton()).toHaveFocus();

    await act(() => stub.engine.history.undo());
    await expect.element(negativeInput()).toHaveValue('blurry');
    expect(liveLayer().negativePrompt).toBe('blurry');
  });
});
