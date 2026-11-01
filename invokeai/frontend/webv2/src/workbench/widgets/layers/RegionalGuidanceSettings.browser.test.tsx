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
vi.mock('./useSelectedModelBase', () => ({ useSelectedModelBase: () => selectedBase }));

type Layer = ComponentProps<typeof RegionalGuidanceSettings>['layer'];

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });
// The real catalog, for copy that names the model. Rendered through its own provider, so it stays out of the
// react-i18next default the instance above registers.
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
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
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

  it.each([
    ['sdxl', true],
    ['flux', false],
  ])('colours and weights regional attention syntax on %s: %s', async (base, readsAttention) => {
    await render(base, createLayer({ positivePrompt: 'a cat+' }));
    const prompt = host!.querySelector('[aria-label="widgets.layers.regionalGuidance.positivePrompt"]')!;

    expect(host!.querySelector('pre') !== null).toBe(readsAttention);
    expect(prompt.matches('[data-prompt-attention-target="true"]')).toBe(readsAttention);
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

describe('RegionalGuidanceSettings on a layout base', () => {
  let stub: StructuralEngineStub;
  const liveLayer = () => getDocumentLayer(stub.document(), 'rg-1') as CanvasRegionalGuidanceLayerContract;

  const LiveSettings = () => {
    const layer = useSyncExternalStore(stub.subscribe, liveLayer);
    return <RegionalGuidanceSettings engine={stub.engine as never} layer={layer} />;
  };

  /** Renders the panel over the real reducer and history, so commits land and undo replays them. */
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

  const kindGroup = () => page.getByRole('radiogroup', { name: 'widgets.layers.regionalGuidance.layoutKind' });
  const pickKind = (kind: 'object' | 'character' | 'text') =>
    act(() => userEvent.click(page.getByText(`widgets.layers.regionalGuidance.layoutKinds.${kind}`, { exact: true })));
  const promptInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.positivePrompt' });
  const negativeInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.negativePrompt' });
  const visibleTextInput = () => page.getByRole('textbox', { name: 'widgets.layers.regionalGuidance.visibleText' });
  /** What the always-present status region announces. */
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
    expect(hasNegativeField()).toBe(false);
    expect(hasAutoNegativeSwitch()).toBe(false);
    expect(alerts()).toEqual([]);
    await act(() => root?.unmount());
    host?.remove();

    await render('sd-1', createLayer());
    expect(kindGroup().query()).toBeNull();
    expect(host!.querySelector('[role="status"]')).toBeNull();
  });

  it('describes the region type by the roles and the prompt field by the box', async () => {
    await renderLive('moss-image');

    expect(descriptionOf(kindGroup().element())).toEqual(['widgets.layers.regionalGuidance.layoutKindHelp']);
    expect(descriptionOf(promptInput().element())).toEqual(['widgets.layers.regionalGuidance.layoutBoxHelp']);
  });

  it('offers no region type or visible text on a base with objects only, keeping a stored text role', async () => {
    await renderLive('ideogram-4', { layout: { kind: 'text' }, negativePrompt: 'blurry' });

    expect(kindGroup().query()).toBeNull();
    expect(visibleTextInput().query()).toBeNull();
    expect(negativeInput().query()).toBeNull();
    // Placed as an object, a text role needs no visible text.
    expect(announced()).toBe('');
    expect(descriptionOf(promptInput().element())).toEqual(['widgets.layers.regionalGuidance.layoutBoxHelp']);
    expect(liveLayer().layout).toEqual({ kind: 'text' });
  });

  it('names FLUX.3 in the help for its box, offering no region type and no negative prompt', async () => {
    await render('flux3', createLayer({ layout: { kind: 'character' }, negativePrompt: 'blurry' }), english);

    // By role alone, so renamed copy cannot hide a control that is shown.
    expect(page.getByRole('radiogroup').query()).toBeNull();
    const textboxes = page.getByRole('textbox').all();
    expect(textboxes).toHaveLength(1);
    await expect
      .element(textboxes[0]!)
      .toHaveAccessibleDescription(
        'FLUX.3 places the rectangle around everything painted on this layer inside the frame, so the shape does not matter and separate areas share one box.'
      );
    expect(alerts()).toEqual([]);
  });

  it('describes the prompt field by the box and a missing prompt together', async () => {
    await renderLive('ideogram-4', { positivePrompt: null });

    expect(announced()).toBe('widgets.layers.regionalGuidance.layoutMissingPrompt');
    expect(descriptionOf(promptInput().element())).toEqual([
      'widgets.layers.regionalGuidance.layoutBoxHelp',
      'widgets.layers.regionalGuidance.layoutMissingPrompt',
    ]);
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

  it('stores visible text across lines, with Enter starting a new line', async () => {
    await renderLive('moss-image', { layout: { kind: 'text' } });

    await act(() => userEvent.type(visibleTextInput(), '一杯温暖{Enter}一刻从容'));
    await act(() => userEvent.tab());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().layout).toEqual({ kind: 'text', text: '一杯温暖\n一刻从容' });
  });

  it('keeps text typed right before a type switch, committing both in order', async () => {
    await renderLive('moss-image', { layout: { kind: 'text' } });

    await act(() => visibleTextInput().fill('EXIT'));
    await pickKind('object');

    expect(stub.commits).toHaveLength(2);
    expect(liveLayer().layout).toEqual({ kind: 'object', text: 'EXIT' });
  });

  it('commits typed visible text when Invoke flushes drafts before the field blurs', async () => {
    await renderLive('moss-image', { layout: { kind: 'text' } });

    await act(() => visibleTextInput().fill('SALE'));
    expect(stub.commits).toHaveLength(0);
    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().layout).toEqual({ kind: 'text', text: 'SALE' });
  });

  it('commits an unsaved caption when Invoke flushes drafts from inside the field (Ctrl+Enter)', async () => {
    await renderLive('moss-image', { positivePrompt: null });
    expect(announced()).toBe('widgets.layers.regionalGuidance.layoutMissingPrompt');
    expect(descriptionOf(promptInput().element())).toEqual([
      'widgets.layers.regionalGuidance.layoutBoxHelp',
      'widgets.layers.regionalGuidance.layoutMissingPrompt',
    ]);

    await act(() => promptInput().fill('a neon sign'));
    expect(stub.commits).toHaveLength(0);
    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().positivePrompt).toBe('a neon sign');
    expect(announced()).toBe('');
    // The field still has focus; leaving it later records nothing new.
    await act(() => userEvent.tab());
    expect(stub.commits).toHaveLength(1);
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

  it('leaves an undone negative prompt undone when drafts flush later', async () => {
    await renderLive('sd-1');

    await act(() => negativeInput().fill('blurry'));
    await act(() => userEvent.tab());
    await act(() => stub.engine.history.undo());
    await expect.element(negativeInput()).toHaveValue('');

    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().negativePrompt).toBeNull();
    expect(stub.engine.history.canRedo()).toBe(true);
  });

  it('commits an unsaved negative prompt on a mask base when drafts are flushed', async () => {
    await renderLive('sd-1');

    await act(() => negativeInput().fill('blurry'));
    await act(() => flushWorkbenchDrafts());

    expect(stub.commits).toHaveLength(1);
    expect(liveLayer().negativePrompt).toBe('blurry');
  });
});
