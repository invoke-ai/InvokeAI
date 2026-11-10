import type { WidgetInstanceContract } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import { getSelectedCfgScale } from './selectedModel';

const generate = (values: Record<string, unknown>): WidgetInstanceContract => ({
  createdAt: '2026-11-03T00:00:00.000Z',
  id: 'generate',
  state: { id: 'generate', label: 'Generate', values, version: 1 },
  typeId: 'generate',
});

const project = (values: Record<string, unknown>) => ({ widgetInstances: { generate: generate(values) } });

describe('getSelectedCfgScale', () => {
  it('reads a stored finite CFG Scale', () => {
    expect(getSelectedCfgScale(project({ cfgScale: 1 }))).toBe(1);
    expect(getSelectedCfgScale(project({ cfgScale: 4.5 }))).toBe(4.5);
  });

  it('reads no CFG Scale when it is absent or not a finite number', () => {
    expect(getSelectedCfgScale(project({}))).toBeNull();
    expect(getSelectedCfgScale({ widgetInstances: {} })).toBeNull();
    for (const cfgScale of ['4', null, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(getSelectedCfgScale(project({ cfgScale })), String(cfgScale)).toBeNull();
    }
  });
});
