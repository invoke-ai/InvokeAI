import type { AuthSession } from '@features/identity';
import type { RegisteredWidget, WidgetCapabilities } from '@workbench/widgetContracts';

import { describe, expect, it } from 'vitest';

import { frontendConfigQueryOptions } from './frontendConfig';
import { resolveWidgetCapabilities } from './widgetCapabilities';
import { getWidgetsForRegion, isWidgetAvailable } from './widgetRegistry';

const singleUser: AuthSession = {
  accountEpoch: 3,
  expiredUserId: null,
  multiuserEnabled: false,
  phase: 'ready',
  sessionExpired: false,
  setupRequired: false,
  strictPasswordChecking: false,
  switchingAccount: false,
  user: null,
};

const member = (isAdmin: boolean, overrides: Partial<AuthSession> = {}): AuthSession => ({
  ...singleUser,
  multiuserEnabled: true,
  user: {
    created_at: '',
    display_name: null,
    email: 'ada@example.com',
    is_active: true,
    is_admin: isAdmin,
    last_login_at: null,
    updated_at: '',
    user_id: 'ada',
  },
  ...overrides,
});

const answered = (canView: boolean | undefined) => ({
  data: canView === undefined ? { show_donation_link: true } : { can_view_server_output: canView },
  fetchStatus: 'idle' as const,
  status: 'success' as const,
});

const granted: WidgetCapabilities = { viewServerOutput: 'granted' };
const denied: WidgetCapabilities = { viewServerOutput: 'denied' };

describe('resolveWidgetCapabilities', () => {
  it('grants only when both the session role and the server allow it', () => {
    expect(resolveWidgetCapabilities(singleUser, answered(true))).toEqual(granted);
    expect(resolveWidgetCapabilities(member(true), answered(true))).toEqual(granted);
    expect(resolveWidgetCapabilities(member(false), answered(true))).toEqual(denied);
    expect(resolveWidgetCapabilities({ ...singleUser, phase: 'unknown' }, answered(true))).toEqual(denied);
    expect(resolveWidgetCapabilities(singleUser, answered(false))).toEqual(denied);
  });

  it('is pending until answered, and denies a failed request or an older server without the field', () => {
    expect(resolveWidgetCapabilities(singleUser, undefined).viewServerOutput).toBe('pending');
    expect(
      resolveWidgetCapabilities(singleUser, { data: undefined, fetchStatus: 'fetching', status: 'pending' })
        .viewServerOutput
    ).toBe('pending');
    expect(resolveWidgetCapabilities(singleUser, { data: undefined, fetchStatus: 'idle', status: 'error' })).toEqual(
      denied
    );
    // A view mounting retries a failed read; while it does, the answer is pending again rather than a refusal.
    expect(
      resolveWidgetCapabilities(singleUser, { data: undefined, fetchStatus: 'fetching', status: 'error' })
        .viewServerOutput
    ).toBe('pending');
    expect(resolveWidgetCapabilities(singleUser, answered(undefined))).toEqual(denied);
  });
});

describe('frontend config account scoping', () => {
  it('reads a separate entry per account lifetime, account and known role', () => {
    const key = (session: AuthSession) => JSON.stringify(frontendConfigQueryOptions(session).queryKey);
    const keys = [
      key(member(true)),
      key(member(true, { accountEpoch: 4 })),
      key(member(false)),
      key({ ...member(true), user: { ...member(true).user!, user_id: 'grace' } }),
      key(singleUser),
    ];

    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe('widget availability', () => {
  const widget = (manifest: Partial<RegisteredWidget['manifest']>) => ({ manifest }) as RegisteredWidget;

  it('hides a capability widget from discovery until granted, leaving other widgets alone', () => {
    const ids = (capabilities: WidgetCapabilities) =>
      getWidgetsForRegion('bottom', capabilities).map((registered) => registered.manifest.id);

    expect(ids(granted)).toContain('server-output');
    expect(ids(denied)).not.toContain('server-output');
    expect(ids({ viewServerOutput: 'pending' })).not.toContain('server-output');
    expect(ids(denied)).toEqual(ids(granted).filter((id) => id !== 'server-output'));
  });

  it('keeps the admin rule separate from capabilities', () => {
    expect(isWidgetAvailable(widget({ requiresCapability: 'viewServerOutput' }), granted)).toBe(true);
    // requiresAdmin still needs a multi-user admin session, which this unit test has none of.
    expect(isWidgetAvailable(widget({ requiresAdmin: true }), granted)).toBe(false);
    expect(isWidgetAvailable(widget({}), denied)).toBe(true);
  });
});
