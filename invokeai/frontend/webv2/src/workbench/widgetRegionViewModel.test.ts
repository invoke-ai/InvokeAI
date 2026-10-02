import type {
  NormalizedWidgetManifest,
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetTypeId,
} from '@workbench/widgetContracts';

import { describe, expect, it, vi } from 'vitest';

import { createWidgetImplementationResource } from './widgetImplementationResource';
import {
  canRemoveItem,
  createWidgetRegionViewModel,
  getWidgetRegionItems,
  isRequiredCenterView,
} from './widgetRegionViewModel';

const TestIcon = () => null;
const TestView = () => null;

const createWidget = (
  overrides: Partial<NormalizedWidgetManifest> & Pick<NormalizedWidgetManifest, 'id' | 'label'>
): RegisteredWidget => ({
  implementation: createWidgetImplementationResource(overrides.id, () => Promise.resolve({ view: TestView })),
  manifest: {
    apiVersion: 1,
    allowMultiple: false,
    allowedRegions: ['left', 'center', 'bottom'],
    failurePolicy: { isolateRenderFailure: true, onRegistrationFailure: 'disable' },
    icon: TestIcon,
    load: () => Promise.resolve({ view: TestView }),
    state: { createInitial: () => ({}), persistence: 'project', version: 1 },
    version: 1,
    ...overrides,
  },
  status: 'enabled',
});

const createInstance = (id: string, typeId: WidgetTypeId): WidgetInstanceContract => ({
  createdAt: '2026-01-01T00:00:00.000Z',
  id,
  state: { id: typeId, label: typeId, values: {}, version: 1 },
  typeId,
});

describe('widget region view model', () => {
  it('separates placed items from available items', () => {
    const viewModel = createWidgetRegionViewModel({
      activeInstanceId: 'alpha',
      instanceIds: ['alpha'],
      region: 'left',
      widgetInstances: { alpha: createInstance('alpha', 'alpha') },
      widgets: [createWidget({ id: 'alpha', label: 'Alpha' }), createWidget({ id: 'beta', label: 'Beta' })],
    });

    expect(viewModel.placedItems.map((item) => item.id)).toEqual(['alpha']);
    expect(viewModel.availableItems.map((item) => item.typeId)).toEqual(['beta']);
    expect(viewModel.activeItem?.id).toBe('alpha');
    expect(viewModel.sortableInstanceIds).toEqual(['alpha']);
    expect(getWidgetRegionItems(viewModel).map((item) => item.label)).toEqual(['Alpha', 'Beta']);
  });

  it('keeps a floated widget in the rail at the slot it docks back to, outside the sortable list', () => {
    const widgets = [createWidget({ id: 'a', label: 'A' }), createWidget({ id: 'b', label: 'B' })];
    const widgetInstances = { 'a:1': createInstance('a:1', 'a'), 'b:1': createInstance('b:1', 'b') };
    const viewModel = createWidgetRegionViewModel({
      activeInstanceId: 'b:1',
      floatingWidgets: {
        'a:1': { returnIndex: 0, returnRegion: 'left' },
        'b:1': { returnIndex: 0, returnRegion: 'right' },
      },
      instanceIds: ['b:1'],
      region: 'left',
      widgetInstances,
      widgets,
    });

    expect(viewModel.placedItems.map((item) => [item.id, item.isFloating ?? false])).toEqual([
      ['a:1', true],
      ['b:1', false],
    ]);
    expect(viewModel.sortableInstanceIds).toEqual(['b:1']);
    expect(viewModel.availableItems.map((item) => item.typeId)).toEqual([]);
  });

  it('orders floating slots by their return index and clamps indices the rail no longer has', () => {
    const widgets = ['a', 'b', 'c', 'd', 'e'].map((id) => createWidget({ id, label: id.toUpperCase() }));
    const widgetInstances = Object.fromEntries(
      widgets.map((widget) => [widget.manifest.id, createInstance(widget.manifest.id, widget.manifest.id)])
    );
    const viewModel = createWidgetRegionViewModel({
      floatingWidgets: {
        // Later-sorted insertions must not displace earlier ones.
        d: { returnIndex: 2, returnRegion: 'left' },
        a: { returnIndex: 0, returnRegion: 'left' },
        // Beyond the rail: appended, as docking would.
        e: { returnIndex: 99, returnRegion: 'left' },
        c: { returnRegion: 'left' },
      },
      instanceIds: ['b'],
      region: 'left',
      widgetInstances,
      widgets,
    });

    expect(viewModel.placedItems.map((item) => item.id)).toEqual(['a', 'b', 'd', 'e', 'c']);
    expect(viewModel.sortableInstanceIds).toEqual(['b']);
  });

  it('filters already placed singleton widget types from available items', () => {
    const viewModel = createWidgetRegionViewModel({
      instanceIds: ['alpha'],
      region: 'left',
      widgetInstances: { alpha: createInstance('alpha', 'alpha') },
      widgets: [createWidget({ id: 'alpha', label: 'Alpha' })],
    });

    expect(viewModel.availableItems).toEqual([]);
  });

  it('keeps allowMultiple widget types available after one placement', () => {
    const viewModel = createWidgetRegionViewModel({
      instanceIds: ['alpha'],
      region: 'left',
      widgetInstances: { alpha: createInstance('alpha', 'alpha') },
      widgets: [createWidget({ allowMultiple: true, id: 'alpha', label: 'Alpha' })],
    });

    expect(viewModel.availableItems).toHaveLength(1);
    expect(viewModel.availableItems[0]).toMatchObject({ allowMultiple: true, isEnabled: false, typeId: 'alpha' });
  });

  it('identifies the last enabled center view as required', () => {
    const viewModel = createWidgetRegionViewModel({
      instanceIds: ['canvas', 'toolbar-tools'],
      region: 'center',
      widgetInstances: {
        canvas: createInstance('canvas', 'canvas'),
        'toolbar-tools': createInstance('toolbar-tools', 'toolbar-tools'),
      },
      widgets: [
        createWidget({ centerPlacement: 'view', id: 'canvas', label: 'Canvas' }),
        createWidget({ centerPlacement: 'toolbar', id: 'toolbar-tools', label: 'Toolbar Tools' }),
      ],
    });
    const canvas = viewModel.placedItems.find((item) => item.id === 'canvas');
    const toolbar = viewModel.placedItems.find((item) => item.id === 'toolbar-tools');

    expect(canvas).toBeDefined();
    expect(toolbar).toBeDefined();
    expect(isRequiredCenterView(canvas!, 1)).toBe(true);
    expect(canRemoveItem(canvas!, viewModel)).toBe(false);
    expect(canRemoveItem(toolbar!, viewModel)).toBe(true);
  });

  it('does not call createInitial while deriving available items', () => {
    const createInitial = vi.fn(() => ({ seeded: true }));
    const viewModel = createWidgetRegionViewModel({
      instanceIds: [],
      region: 'left',
      widgetInstances: {},
      widgets: [
        createWidget({
          id: 'alpha',
          label: 'Alpha',
          state: { createInitial, persistence: 'project', version: 1 },
        }),
      ],
    });

    expect(createInitial).not.toHaveBeenCalled();
    expect(viewModel.availableItems[0]).not.toHaveProperty('initialValues');
  });
});
