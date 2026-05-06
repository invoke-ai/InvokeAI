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
  it('reports attention weights as supported for SDXL', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: '(face:1.2)',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'attention-support')).toMatchObject({
      label: 'Weights OK',
      severity: 'ok',
    });
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
      label: 'Missing 1',
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
      label: 'Wildcard error',
      severity: 'error',
    });
    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-missing')).toBeUndefined();
  });

  it('does not count available wildcards as prompt references', () => {
    const diagnostics = getPromptDiagnostics({
      prompt: 'portrait',
      modelBase: 'sdxl',
      wildcards,
      wildcardIndexErrorCount: 0,
      dynamicPromptCount: 1,
      dynamicPromptMode: 'random',
    });

    expect(diagnostics.find((diagnostic) => diagnostic.code === 'wildcards-available')).toMatchObject({
      label: 'Wildcards',
      severity: 'info',
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
      label: 'Random 1',
      severity: 'ok',
    });
  });

  it('reports random refresh on invoke', () => {
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
      label: 'Random 1/run',
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
      label: 'Cycle 1',
      severity: 'ok',
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
      label: 'Dynamic error',
      severity: 'error',
      description: 'Dynamic prompt parser error: Could not parse prompt',
    });
  });
});
