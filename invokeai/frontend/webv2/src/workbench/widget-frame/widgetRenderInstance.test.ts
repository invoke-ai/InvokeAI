import type { WidgetInstanceContract } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import { areProjectWidgetRenderInstancesEqual } from './widgetRenderInstance';

const instance: WidgetInstanceContract = {
  createdAt: '2026-01-01T00:00:00.000Z',
  id: 'generate',
  state: { id: 'generate', label: 'Generate', values: {}, version: 1 },
  title: 'Generate',
  typeId: 'generate',
};

describe('areProjectWidgetRenderInstancesEqual', () => {
  it('treats same instance metadata in different projects as different render inputs', () => {
    expect(
      areProjectWidgetRenderInstancesEqual(
        { instance, projectId: 'project-a' },
        { instance: { ...instance }, projectId: 'project-b' }
      )
    ).toBe(false);
  });
});
