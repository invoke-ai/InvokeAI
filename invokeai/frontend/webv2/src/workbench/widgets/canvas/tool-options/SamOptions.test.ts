import type { SamSessionSnapshot } from '@workbench/canvas-engine/engineStores';
import type { ComponentProps } from 'react';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import {
  getSamPanelViewModel,
  getSamActionEligibility,
  getSamErrorTranslationKey,
  getSamStatusTranslationKey,
  SAM_AUTO_FIT_COLUMNS,
  SamModeToggle,
  SamProcessFeedback,
} from './SamOptions';

const englishCatalogModules = import.meta.glob('../../../../../public/locales/en.json', {
  eager: true,
  import: 'default',
});
const en = Object.values(englishCatalogModules)[0] as {
  widgets: { layers: { selectObject: Record<string, string> } };
};

const snapshot = (overrides: Partial<SamSessionSnapshot> = {}): SamSessionSnapshot => ({
  applyPolygonRefinement: false,
  autoProcess: false,
  error: null,
  hasPreview: false,
  input: { bbox: null, excludePoints: [], includePoints: [], type: 'visual' },
  invert: false,
  isolatedPreview: true,
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
  it('derives visual counts, bbox state, and a localized generation-area source summary', () => {
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
          sourceRect: { height: 768, width: 1024, x: 64, y: 32 },
        }),
        (width, height) => `Área de generación ${width} por ${height}`
      )
    ).toEqual({
      bboxActive: true,
      excludeCount: 1,
      includeCount: 2,
      sourceSummary: 'Área de generación 1024 por 768',
    });
  });

  it('returns zero visual counts for prompt mode', () => {
    expect(
      getSamPanelViewModel(snapshot({ input: { prompt: 'cat', type: 'prompt' } }), (w, h) => `${w} × ${h}`)
    ).toEqual({
      bboxActive: false,
      excludeCount: 0,
      includeCount: 0,
      sourceSummary: '20 × 20',
    });
  });
});

describe('SamModeToggle', () => {
  it('uses available inline width instead of viewport breakpoints', () => {
    expect(SAM_AUTO_FIT_COLUMNS).toBe('repeat(auto-fit, minmax(min(100%, 11rem), 1fr))');
  });

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

describe('SamProcessFeedback', () => {
  it.each([
    'ready',
    'scheduled',
    'preparing-composite',
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
      createElement(SamProcessFeedback, {
        error: null,
        errorText: null,
        statusText: 'Rendering object preview…',
      })
    );

    expect(markup).toContain('role="status"');
    expect(markup).toContain('aria-live="polite"');
    expect(markup).toContain('Rendering object preview…');
    expect(markup).not.toContain('role="alert"');
  });

  it('server-renders a known localized error as the assertive primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: { code: 'decode' },
        errorText: 'The object preview could not be decoded.',
        statusText: 'Select Object needs attention.',
      })
    );

    expect(markup).toContain('role="alert"');
    expect(markup).toContain('aria-live="assertive"');
    expect(markup).toContain('The object preview could not be decoded.');
  });

  it('server-renders unknown diagnostics only after a localized primary message', () => {
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: { code: 'unknown', detail: 'GPU worker disconnected' },
        errorText: 'Select Object could not finish.',
        statusText: 'Select Object needs attention.',
      })
    );

    expect(markup.indexOf('Select Object could not finish.')).toBeLessThan(markup.indexOf('GPU worker disconnected'));
    expect(markup).toContain('role="alert"');
  });

  it('server-renders a localized queue error before its backend exception type', () => {
    const errorText = en.widgets.layers.selectObject.errorQueue;
    const markup = renderToStaticMarkup(
      createElement(SamProcessFeedback, {
        error: { code: 'queue', detail: 'AttributeError' },
        errorText,
        statusText: 'Select Object needs attention.',
      })
    );

    expect(markup.indexOf(errorText)).toBeLessThan(markup.indexOf('AttributeError'));
    expect(markup).toContain('role="alert"');
  });
});
