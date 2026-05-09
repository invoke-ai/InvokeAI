import { describe, expect, it } from 'vitest';

import { getPromptDiagnostics } from './diagnostics';

const wildcards = [
  {
    token: '__camera/lens__',
    path: 'camera/lens',
    label: 'lens',
    file_type: 'txt' as const,
    value_count: 2,
    samples: ['50mm', '85mm'],
  },
];

describe('prompt workbench diagnostics', () => {
  it('hides supported attention weight status for SDXL', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: '(face:1.2)',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code.startsWith('attention'))).toBeUndefined();
  });

  it('warns when attention syntax is used with FLUX-like models', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: '(face:1.2)',
      modelBase: 'flux',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'attention-unsupported')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.weightsLiteralLabel' },
      severity: 'warning',
    });
  });

  it('reports missing wildcard references', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait with __missing__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-missing')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.missingLabel', options: { count: 1 } },
      severity: 'error',
    });
  });

  it('reports unavailable wildcard index instead of false missing references', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait with __camera/lens__',
      modelBase: 'sdxl',
      wildcards: [],
      wildcardIndexUnavailable: true,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-unavailable')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.wildcardErrorLabel' },
      severity: 'error',
    });
    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-missing')).toBeUndefined();
  });

  it('hides generic available wildcard status when the prompt has no references', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-available')).toBeUndefined();
    expect(
      diagnostics.find((diagnostic) => diagnostic.label.key === 'promptWorkbench.diagnostics.wildcardsLabel')
    ).toBeUndefined();
  });

  it('reports referenced wildcards when present', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-found')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.wildcardsFoundLabel', options: { count: 1 } },
      severity: 'ok',
    });
  });

  it('keeps wildcard index error warnings visible', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 2,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-index-errors')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.indexErrorsLabel', options: { count: 2 } },
      severity: 'warning',
    });
  });

  it('reports dynamic prompt count by mode', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.behavior.randomImageShort' },
      severity: 'ok',
    });
  });

  it('reports random refresh per image', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
      dynamicPromptRandomRefreshMode: 'per_image',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.behavior.randomImageShort' },
      severity: 'ok',
    });
  });

  it('reports random refresh per invoke', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
      dynamicPromptRandomRefreshMode: 'per_enqueue',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.behavior.randomInvokeShort' },
      severity: 'ok',
    });
  });

  it('reports locked preview random refresh', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
      dynamicPromptRandomRefreshMode: 'manual',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.behavior.randomPreviewLabel' },
      severity: 'ok',
    });
  });

  it('reports cyclic wildcard prompts as deterministic', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __@camera/lens__',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
      dynamicPromptRandomRefreshMode: 'per_enqueue',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.dynamicCycleLabel', options: { count: 1 } },
      severity: 'ok',
    });
  });

  it('reports mixed cyclic and random dynamic syntax', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait __@camera/lens__ {warm|cool}',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 2,
      dynamicPromptMode: 'random',
      dynamicPromptRandomRefreshMode: 'per_enqueue',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-active')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.dynamicMixedLabel' },
      severity: 'ok',
      description: { key: 'promptWorkbench.diagnostics.dynamicMixedDesc' },
    });
  });

  it('surfaces dynamic prompt parser errors in the tooltip description', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait {broken',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
      dynamicPromptError: 'Could not parse prompt',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'dynamic-error')).toMatchObject({
      label: { key: 'promptWorkbench.diagnostics.dynamicErrorLabel' },
      severity: 'error',
      description: {
        key: 'promptWorkbench.diagnostics.dynamicErrorDesc',
        options: { error: 'Could not parse prompt' },
      },
    });
  });
});
