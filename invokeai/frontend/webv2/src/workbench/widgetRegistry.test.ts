import { describe, expect, it } from 'vitest';

import type { WidgetManifest } from './types';

import { registerFirstPartyWidgets, registerWidgets } from './widgetRegistry';

const TestIcon = () => null;
const TestView = () => null;

const createManifest = (overrides: Partial<WidgetManifest> = {}): WidgetManifest => ({
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: TestIcon,
  id: 'vendor.widget',
  label: 'Test Widget',
  labelText: 'Test Widget',
  version: 1,
  view: TestView,
  ...overrides,
});

describe('widget registry', () => {
  it('registers first-party widget manifests without icon validation failures', () => {
    const widgets = registerFirstPartyWidgets();

    expect(widgets).toHaveLength(14);
    expect(widgets.flatMap((widget) => widget.failure ?? [])).toEqual([]);
    expect(widgets.every((widget) => widget.status === 'enabled')).toBe(true);
  });

  it('normalizes default state and api version', () => {
    const [widget] = registerWidgets([createManifest({ state: undefined })]);

    expect(widget.status).toBe('enabled');
    expect(widget.manifest.apiVersion).toBe(1);
    expect(widget.manifest.state).toMatchObject({ persistence: 'project', version: 1 });
    expect(widget.manifest.state.createInitial()).toEqual({});
  });

  it('disables invalid icon manifests', () => {
    const [widget] = registerWidgets([createManifest({ icon: 'invalid' as unknown as WidgetManifest['icon'] })]);

    expect(widget.status).toBe('disabled');
    expect(widget.failure?.message).toContain('must provide an icon component');
  });

  it('fails closed for empty regions', () => {
    const [widget] = registerWidgets([createManifest({ allowedRegions: [] })]);

    expect(widget.status).toBe('disabled');
    expect(widget.failure?.message).toContain('must declare at least one allowed region');
  });

  it('fails closed for renderable regions without a view', () => {
    const [widget] = registerWidgets([createManifest({ view: undefined })]);

    expect(widget.status).toBe('disabled');
    expect(widget.failure?.message).toContain('does not include manifest.view');
  });

  it('honors hidden failure policy for invalid manifests', () => {
    const [widget] = registerWidgets([
      createManifest({
        failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'hide' },
        view: undefined,
      }),
    ]);

    expect(widget.status).toBe('hidden');
    expect(widget.failure).toBeDefined();
  });

  it('rejects unstable ids and unsupported api versions', () => {
    const [badId, badApiVersion] = registerWidgets([
      createManifest({ id: 'bad id' }),
      createManifest({ apiVersion: 2 as unknown as WidgetManifest['apiVersion'], id: 'vendor.bad-api' }),
    ]);

    expect(badId.status).toBe('disabled');
    expect(badId.failure?.message).toContain('stable non-empty string id');
    expect(badApiVersion.status).toBe('disabled');
    expect(badApiVersion.failure?.message).toContain('unsupported apiVersion');
  });
});
