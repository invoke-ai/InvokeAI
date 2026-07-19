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
