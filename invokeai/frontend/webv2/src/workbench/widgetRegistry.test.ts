import { ImageUpscaleIcon } from 'lucide-react';
import { describe, expect, it } from 'vitest';

import type { WidgetManifest } from './types';

import { getWidgetHosts, registerFirstPartyWidgets, registerWidgets } from './widgetRegistry';
import { upscaleWidgetManifest } from './widgets/upscale';

const TestIcon = () => null;
const TestView = () => null;

const createManifest = (overrides: Partial<WidgetManifest> = {}): WidgetManifest => ({
  allowMultiple: false,
  allowedRegions: ['right'],
  failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
  icon: TestIcon,
  id: 'vendor.widget',
  label: 'Test Widget',
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

  it('uses the image-upscale icon for Upscale', () => {
    expect(upscaleWidgetManifest.icon).toBe(ImageUpscaleIcon);
  });

  it('normalizes default state and api version', () => {
    const [widget] = registerWidgets([createManifest({ state: undefined })]);

    expect(widget.status).toBe('enabled');
    expect(widget.manifest.apiVersion).toBe(1);
    expect(widget.manifest.state).toMatchObject({ persistence: 'project', version: 1 });
    expect(widget.manifest.state.createInitial()).toEqual({});
  });

  it('exposes enabled widget singleton hosts', () => {
    expect(getWidgetHosts().map((widget) => widget.manifest.id)).toContain('workflow');
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
