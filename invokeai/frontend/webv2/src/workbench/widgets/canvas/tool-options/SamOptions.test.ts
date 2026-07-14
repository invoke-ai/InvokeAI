import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createInstance } from 'i18next';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { I18nextProvider } from 'react-i18next';
import { describe, expect, it, vi } from 'vitest';

import {
  getSamPanelViewModel,
  getSamActionEligibility,
  getSamActionHandlers,
  getSamErrorTranslationKey,
  getSamStatusTranslationKey,
  keepSamImageIntermediate,
  SamPromptBody,
  SamModeToggle,
  SamOptionsBar,
  SamStatusSlot,
  SamVisualInput,
} from './SamOptions';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const enCatalog = Object.values(englishCatalogModules)[0] as {
  common: Record<string, string>;
  widgets: { layers: { selectObject: Record<string, string> } };
};
const en = enCatalog;
const testI18n = createInstance();
await testI18n.init({
  initImmediate: false,
  lng: 'en',
  resources: { en: { translation: enCatalog } },
  showSupportNotice: false,
});

const snapshot = (overrides: Partial<SamSessionSnapshot> = {}): SamSessionSnapshot => ({
  applyPolygonRefinement: false,
  autoProcess: false,
  error: null,
  hasPreview: false,
  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
  invert: false,
  isolatedPreview: true,
  layerName: 'Layer 1',
  layerType: 'raster',
  model: 'segment-anything-2-large',
  pointLabel: 'include',
  sourceRect: { height: 20, width: 20, x: 0, y: 0 },
  status: 'ready',
  ...overrides,
});

describe('getSamActionEligibility', () => {
  it('requires valid visual or prompt input before processing', () => {
    expect(getSamActionEligibility(snapshot())).toMatchObject({ canProcess: false });
    expect(
      getSamActionEligibility(
        snapshot({ input: { bbox: null, excludePoints: [], includePoints: [{ x: 1, y: 2 }], type: 'visual' } })
      )
    ).toMatchObject({ canProcess: true });
    expect(getSamActionEligibility(snapshot({ input: { prompt: '  object  ', type: 'prompt' } }))).toMatchObject({
      canProcess: true,
    });
  });

  it('permits Apply and Save only for a current ready preview', () => {
    expect(getSamActionEligibility(snapshot({ hasPreview: true }))).toMatchObject({ canApply: true, canSave: true });
    expect(getSamActionEligibility(snapshot({ hasPreview: true, status: 'processing-sam' }))).toMatchObject({
      canApply: false,
      canProcess: false,
      canSave: false,
    });
    expect(getSamActionEligibility(snapshot({ hasPreview: false }))).toMatchObject({
      canApply: false,
      canSave: false,
    });
    expect(
      getSamActionEligibility(
        snapshot({ error: { code: 'unknown', detail: 'commit failed' }, hasPreview: true, status: 'error' })
      )
    ).toMatchObject({
      canApply: true,
      canSave: true,
    });
  });

  it('keeps Cancel available while processing and Reset available for every active session', () => {
    expect(getSamActionEligibility(snapshot({ status: 'processing-sam' }))).toMatchObject({
      canCancel: true,
      canReset: true,
    });
    expect(getSamActionEligibility(snapshot({ error: { code: 'not-ready' }, status: 'error' }))).toMatchObject({
      canReset: true,
    });
    expect(getSamActionEligibility(snapshot({ status: 'ready' }))).toMatchObject({ canReset: true });
  });

  it('disables every mutating control while committing and keeps Cancel available', () => {
    const committing = snapshot({ hasPreview: true, status: 'committing' as never });

    expect(getSamActionEligibility(committing)).toEqual({
      canApply: false,
      canCancel: true,
      canEditInputs: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
  });

  it('disables mutating actions under an external interaction lock but preserves Cancel', () => {
    expect(getSamActionEligibility(snapshot({ hasPreview: true }), true)).toEqual({
      canApply: false,
      canCancel: true,
      canEditInputs: false,
      canProcess: false,
      canReset: false,
      canSave: false,
    });
  });
});

describe('getSamPanelViewModel', () => {
  it('derives visual counts, bbox state, and a localized layer source label', () => {
    expect(
      getSamPanelViewModel(
        snapshot({
          input: {
            bbox: { height: 8, width: 12, x: 1, y: 2 },
            excludePoints: [{ x: 3, y: 4 }],
            includePoints: [
              { x: 1, y: 2 },
              { x: 2, y: 3 },
            ],
            type: 'visual',
          },
          layerName: 'Subject',
          sourceRect: { height: 768, width: 1024, x: 64, y: 32 },
        }),
        (layerName, width, height) => `${layerName} at ${width} by ${height}`
      )
    ).toEqual({
      bboxActive: true,
      excludeCount: 1,
      includeCount: 2,
      sourceLabel: 'Subject at 1024 by 768',
    });
  });

  it('returns zero visual counts for prompt mode', () => {
    expect(
      getSamPanelViewModel(
        snapshot({ input: { prompt: 'cat', type: 'prompt' } }),
        (layerName, w, h) => `${layerName} ${w} × ${h}`
      )
    ).toEqual({
      bboxActive: false,
      excludeCount: 0,
      includeCount: 0,
      sourceLabel: 'Layer 1 20 × 20',
    });
  });
});

describe('SamModeToggle', () => {
  it('uses ordinary pressed buttons without incomplete tab relationships', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamModeToggle, {
          disabled: false,
          mode: 'visual',
          onPrompt: () => undefined,
          onVisual: () => undefined,
          promptLabel: 'Prompt',
          visualLabel: 'Visual',
        })
      )
    );

    expect(markup).toContain('aria-label="Selection mode"');
    expect(markup).toContain('aria-pressed="true"');
    expect(markup).toContain('aria-pressed="false"');
    expect(markup).not.toContain('role="tab"');
    expect(markup).not.toContain('aria-controls');
  });
});

describe('SamStatusSlot', () => {
  it.each([
    'ready',
    'scheduled',
    'preparing-source',
    'uploading',
    'processing-sam',
    'rendering-preview',
    'committing',
    'error',
  ] as const)('maps %s to localized user-facing copy without exposing the identifier', (status) => {
    const key = getSamStatusTranslationKey(status).split('.').at(-1);
    const copy = key ? en.widgets.layers.selectObject[key] : undefined;
    expect(copy).toBeTypeOf('string');
    expect(copy).not.toContain(status);
  });

  it.each([
    'invalid',
    'not-ready',
    'empty',
    'upload',
    'queue',
    'no-output',
    'reconcile',
    'output-dimension',
    'decode',
    'locked',
    'unknown',
  ] as const)('maps %s errors to localized primary copy', (code) => {
    const key = getSamErrorTranslationKey(code).split('.').at(-1);
    const copy = key ? en.widgets.layers.selectObject[key] : undefined;
    expect(copy).toBeTypeOf('string');
    expect(copy).not.toBe(code);
  });

  it('server-renders a localized phase with polite status semantics', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamStatusSlot, {
          error: null,
          errorText: null,
          isBusy: true,
          statusText: 'Rendering object preview…',
          technicalDetailsLabel: 'Technical details',
        })
      )
    );

    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).toContain('Rendering object preview…');
    expect(markup).not.toContain('role="alert"');
  });

  it('reserves an empty polite live region when ready without an error', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamStatusSlot, {
          error: null,
          errorText: null,
          isBusy: false,
          statusText: 'Ready',
          technicalDetailsLabel: 'Technical details',
        })
      )
    );

    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).not.toContain('Ready');
    expect(markup).not.toContain('role="alert"');
  });

  it('server-renders a known localized error as the assertive primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamStatusSlot, {
          error: { code: 'decode' },
          errorText: 'The object preview could not be decoded.',
          isBusy: false,
          statusText: 'Select Object needs attention.',
          technicalDetailsLabel: 'Technical details',
        })
      )
    );

    expect(markup).toContain('role="alert"');
    expect(markup).toContain('aria-live="assertive"');
    expect(markup).toContain('The object preview could not be decoded.');
  });

  it('server-renders unknown diagnostics only after a localized primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamStatusSlot, {
          error: { code: 'unknown', detail: 'GPU worker disconnected' },
          errorText: 'Select Object could not finish.',
          isBusy: false,
          statusText: 'Select Object needs attention.',
          technicalDetailsLabel: 'Technical details',
        })
      )
    );

    expect(markup).not.toContain('title=');
    expect(markup).toContain('aria-label="Technical details"');
    expect(markup).toContain('tabindex="0"');
    expect(markup).toContain('data-scope="tooltip"');
    expect(markup).not.toContain('>GPU worker disconnected<');
    expect(markup).toContain('role="alert"');
  });

  it('server-renders a localized queue error before its backend exception type', () => {
    const errorText = en.widgets.layers.selectObject.errorQueue;
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamStatusSlot, {
          error: { code: 'queue', detail: 'AttributeError' },
          errorText,
          isBusy: false,
          statusText: 'Select Object needs attention.',
          technicalDetailsLabel: 'Technical details',
        })
      )
    );

    expect(markup).not.toContain('title=');
    expect(markup).toContain('aria-label="Technical details"');
    expect(markup).toContain('data-scope="tooltip"');
    expect(markup).not.toContain('>AttributeError<');
    expect(markup).toContain('role="alert"');
  });
});

describe('single-row SAM inputs', () => {
  it('uses bar-scale copy including the settings popover and bbox indicator labels', () => {
    expect(en.widgets.layers.selectObject.model).toBe('Model');
    expect(en.widgets.layers.selectObject.refine).toBe('Refine');
    expect(en.widgets.layers.selectObject.autoProcess).toBe('Auto Process');
    expect(en.widgets.layers.selectObject.isolatedPreview).toBe('Isolated Preview');
    expect(en.widgets.layers.selectObject.settings).toBe('Select Object settings');
    expect(en.widgets.layers.selectObject.bbox).toBe('BBox');
    expect(en.widgets.layers.selectObject.technicalDetails).toBe('Technical details');
    expect(en.widgets.layers.selectObject.sourceLayerLabel).toContain('{{name}}');
    expect(en.widgets.layers.selectObject.sourceLayerLabel).toContain('{{width}}');
    expect(en.widgets.layers.selectObject.saveAs_selection).toBe('Selection');
    expect(en.widgets.layers.selectObject.includeCount).toBe('Include {{count}}');
    expect(en.widgets.layers.selectObject.excludeCount).toBe('Exclude {{count}}');
    expect(en.widgets.layers.selectObject).not.toHaveProperty('autoProcessCompact');
    expect(en.widgets.layers.selectObject).not.toHaveProperty('isolatedPreviewCompact');
  });

  it('server-renders Prompt as a one-line accessible input', () => {
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(SamPromptBody, { disabled: false, prompt: 'cat', onChange: () => undefined })
      )
    );

    expect(markup).toContain('<input');
    expect(markup).toContain('aria-label=');
    expect(markup).toContain('aria-describedby="sam-prompt-guidance"');
    expect(markup).toContain('id="sam-prompt-guidance"');
    expect(markup).toContain('placeholder=');
    expect(markup).not.toContain('<textarea');
  });

  it('server-renders the visual controls as one labeled group without permanent guidance', () => {
    const session = snapshot({
      input: {
        bbox: { height: 8, width: 12, x: 1, y: 2 },
        excludePoints: [{ x: 3, y: 4 }],
        includePoints: [{ x: 1, y: 2 }],
        type: 'visual',
      },
    });
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(
          I18nextProvider,
          { i18n: testI18n },
          createElement(SamVisualInput, {
            disabled: false,
            onExclude: () => undefined,
            onInclude: () => undefined,
            pointLabel: session.pointLabel,
            viewModel: getSamPanelViewModel(session, (layerName, width, height) => `${layerName} ${width} × ${height}`),
          })
        )
      )
    );

    expect(markup).toContain('role="group"');
    expect(markup).toContain('aria-pressed="true"');
    expect(markup).not.toContain('title=');
    expect(markup).toContain('aria-describedby="sam-visual-guidance"');
    expect(markup).toContain('id="sam-visual-guidance"');
    expect(markup).toContain('Include 1');
    expect(markup).toContain('Exclude 1');
    expect(markup).toContain('Bounding box active');
    expect(markup).not.toContain('>widgets.layers.selectObject.pointType<');
  });
});

describe('getSamActionHandlers', () => {
  it('keeps adopted SAM images intermediate', async () => {
    await expect(keepSamImageIntermediate('sam-selection.png')).resolves.toBeUndefined();
  });

  it('wires every bar action and save target to the engine', () => {
    const engine = {
      applySelectObjectSession: vi.fn(),
      cancelSelectObjectSession: vi.fn(),
      processSelectObjectSession: vi.fn(),
      resetSelectObjectSession: vi.fn(),
      saveSelectObjectSession: vi.fn(),
    };
    const actions = getSamActionHandlers(engine as never);

    actions.process();
    actions.reset();
    actions.apply();
    actions.save('raster');
    actions.save('control');
    actions.save('inpaint_mask');
    actions.save('regional_guidance');
    actions.save('selection');
    actions.cancel();

    expect(engine.processSelectObjectSession).toHaveBeenCalledOnce();
    expect(engine.resetSelectObjectSession).toHaveBeenCalledOnce();
    expect(engine.applySelectObjectSession).toHaveBeenCalledOnce();
    expect(engine.applySelectObjectSession).toHaveBeenCalledWith(keepSamImageIntermediate);
    expect(engine.processSelectObjectSession.mock.invocationCallOrder[0]).toBeLessThan(
      engine.resetSelectObjectSession.mock.invocationCallOrder[0] as number
    );
    expect(engine.resetSelectObjectSession.mock.invocationCallOrder[0]).toBeLessThan(
      engine.applySelectObjectSession.mock.invocationCallOrder[0] as number
    );
    expect(engine.saveSelectObjectSession).toHaveBeenCalledWith('raster', keepSamImageIntermediate);
    expect(engine.saveSelectObjectSession).toHaveBeenCalledWith('control', keepSamImageIntermediate);
    expect(engine.saveSelectObjectSession).toHaveBeenCalledWith('inpaint_mask', keepSamImageIntermediate);
    expect(engine.saveSelectObjectSession).toHaveBeenCalledWith('regional_guidance', keepSamImageIntermediate);
    expect(engine.saveSelectObjectSession).toHaveBeenCalledWith('selection', keepSamImageIntermediate);
    expect(engine.cancelSelectObjectSession).toHaveBeenCalledOnce();
  });
});

describe('SamOptionsBar', () => {
  it('server-renders one row of controls in stable order without panel slots', () => {
    const engine = {
      applySelectObjectSession: vi.fn(),
      cancelSelectObjectSession: vi.fn(),
      processSelectObjectSession: vi.fn(),
      resetSelectObjectSession: vi.fn(),
      saveSelectObjectSession: vi.fn(),
      updateSelectObjectSession: vi.fn(),
    };
    const markup = renderToStaticMarkup(
      createElement(
        ChakraProvider,
        { value: system } as ComponentProps<typeof ChakraProvider>,
        createElement(
          I18nextProvider,
          { i18n: testI18n },
          createElement(SamOptionsBar, {
            engine: engine as never,
            session: snapshot({
              hasPreview: true,
              input: { bbox: null, excludePoints: [], includePoints: [{ x: 4, y: 5 }], type: 'visual' },
              sourceRect: { height: 768, width: 1024, x: 0, y: 0 },
            }),
          })
        )
      )
    );

    expect(markup).not.toContain('data-slot="header"');
    expect(markup).not.toContain('data-slot="body"');
    expect(markup).not.toContain('data-slot="footer"');
    expect(markup).not.toContain('data-operation=');
    expect(markup).toContain('Layer 1 · Raster layer · 1024 × 768');
    expect(markup).toContain('aria-label="Select Object settings"');
    expect(markup).toContain('aria-label="Save As"');
    expect(markup.indexOf('>Visual<')).toBeLessThan(markup.indexOf('Include 1'));
    expect(markup.indexOf('Include 1')).toBeLessThan(markup.indexOf('>Invert<'));
    expect(markup.indexOf('>Invert<')).toBeLessThan(markup.indexOf('aria-label="Select Object settings"'));
    expect(markup.indexOf('aria-label="Select Object settings"')).toBeLessThan(markup.indexOf('>Process<'));
    expect(markup.indexOf('>Process<')).toBeLessThan(markup.indexOf('>Reset<'));
    expect(markup.indexOf('>Reset<')).toBeLessThan(markup.indexOf('>Apply<'));
    expect(markup.indexOf('>Apply<')).toBeLessThan(markup.indexOf('aria-label="Save As"'));
    expect(markup.indexOf('aria-label="Save As"')).toBeLessThan(markup.indexOf('>Cancel<'));
  });
});
