import type { ExternalProviderStatus } from 'services/api/types';
import { describe, expect, it } from 'vitest';

import { getExternalProviderStatusBadgeInfo } from './externalProviderStatusUtils';

const buildStatus = (overrides: Partial<ExternalProviderStatus> = {}): ExternalProviderStatus => ({
  provider_id: 'openai',
  configured: false,
  message: null,
  ...overrides,
});

describe('getExternalProviderStatusBadgeInfo', () => {
  it('marks configured providers as configured', () => {
    const badgeInfo = getExternalProviderStatusBadgeInfo(buildStatus({ configured: true }));

    expect(badgeInfo.labelKey).toBe('settings.externalProviderConfigured');
    expect(badgeInfo.tooltipKey).toBeNull();
    expect(badgeInfo.tooltipMessage).toBeNull();
    expect(badgeInfo.colorScheme).toBe('green');
  });

  it('adds hint when provider is not configured', () => {
    const badgeInfo = getExternalProviderStatusBadgeInfo(buildStatus());

    expect(badgeInfo.labelKey).toBe('settings.externalProviderNotConfigured');
    expect(badgeInfo.tooltipKey).toBe('settings.externalProviderNotConfiguredHint');
    expect(badgeInfo.tooltipMessage).toBeNull();
    expect(badgeInfo.colorScheme).toBe('warning');
  });

  it('prefers status messages when present', () => {
    const badgeInfo = getExternalProviderStatusBadgeInfo(buildStatus({ message: 'Missing key' }));

    expect(badgeInfo.tooltipKey).toBeNull();
    expect(badgeInfo.tooltipMessage).toBe('Missing key');
  });
});
