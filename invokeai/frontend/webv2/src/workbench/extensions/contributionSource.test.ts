import type { WidgetContributionSource } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import {
  areWidgetContributionSourcesEqual,
  getWidgetContributionRegistrationKey,
  getWidgetContributionSourceKey,
} from './contributionSource';

const source = (overrides: Partial<WidgetContributionSource> = {}): WidgetContributionSource => ({
  instanceId: 'generate',
  projectId: 'project-1',
  region: 'left',
  typeId: 'generate',
  ...overrides,
});

describe('areWidgetContributionSourcesEqual', () => {
  it('matches widget contribution sources by project, region, type, and instance', () => {
    expect(areWidgetContributionSourcesEqual(source(), source())).toBe(true);
    expect(areWidgetContributionSourcesEqual(source(), source({ projectId: 'project-2' }))).toBe(false);
    expect(areWidgetContributionSourcesEqual(source(), source({ instanceId: 'generate:2' }))).toBe(false);
    expect(areWidgetContributionSourcesEqual(source(), null)).toBe(false);
  });
});

describe('getWidgetContributionRegistrationKey', () => {
  it('creates stable replacement keys per command id and source', () => {
    expect(getWidgetContributionRegistrationKey('generate.run', source())).toBe(
      getWidgetContributionRegistrationKey('generate.run', source())
    );
    expect(getWidgetContributionRegistrationKey('generate.run', source())).not.toBe(
      getWidgetContributionRegistrationKey('generate.run', source({ projectId: 'project-2' }))
    );
    expect(getWidgetContributionRegistrationKey('generate.run', undefined)).not.toBe(
      getWidgetContributionRegistrationKey('generate.run', source())
    );
  });

  it('keeps source keys stable without depending on object identity', () => {
    expect(getWidgetContributionSourceKey(source())).toBe(getWidgetContributionSourceKey({ ...source() }));
  });
});
