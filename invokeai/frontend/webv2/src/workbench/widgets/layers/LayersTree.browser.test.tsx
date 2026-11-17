/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasEngine, CanvasNodeContract, PreparedDocumentEdit } from '@workbench/canvas-engine/api';
import type { CanvasOperationState } from '@workbench/canvas-operations/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import {
  compileDocumentLeaves,
  createDocumentModel,
  getDocumentIndex,
  getDocumentLayer,
  getDocumentLeaves,
  isNodeHidden,
} from '@workbench/canvas-engine/api';
import {
  groupContract,
  layerContract,
  stacksFrom,
} from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { attachCanvasOperations } from '@workbench/canvas-operations/operationAccess';
import { createEmptyCanvasDocument } from '@workbench/canvasMigration';
import { applyCanvasProjectMutation } from '@workbench/canvasProjectMutations';
import {
  clearLayerPanelStates,
  readLayerPanelState,
  reconcileLayerPanelStates,
  setLayerPanelFilter,
  toggleLayerStackCollapsed,
  useLayerPanelState,
} from '@workbench/layerPanelState';
import { executeCanvasHotkeyCommand, type CanvasHotkeyContext } from '@workbench/widgets/canvas/canvasHotkeyCommands';
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act, useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { LayerRowCommands } from './layerRowCommands';
import type * as LayerStackHeaderModule from './LayerStackHeader';

import { createFakeAdjustmentStatus } from './AdjustmentStatusIcon.testing';
import { clearLayerChildSelection, getLayerChildSelection } from './layerChildSelection';
import { getLayerRowCommits, resetLayerRowCommits } from './layerPanelDiagnostics';
import { LAYER_PANEL_DEGRADE_THRESHOLD, LAYER_ROW_HEIGHT_PX } from './layerPanelRows';
import { clearLayerPropertiesRequest } from './layerPropertiesRequestStore';
import { LayersTree, type LayersTreeEngine } from './LayersTree';
import { buildLayerStackRows } from './layerTreeRows';

vi.mock('./ControlLayerStatusIcon', () => ({ ControlLayerStatusIcon: () => null }));
vi.mock('./LayerThumbnail', () => ({ LayerThumbnail: () => <span data-testid="thumbnail" /> }));
const realStackHeader = vi.hoisted(() => ({ on: false, exporting: false }));
vi.mock('./useLayerStackActions', async () => {
  const { getStackActions } = await import('./layerStackActions');
  const { PlusIcon } = await import('lucide-react');
  const labels = {
    exportPsd: 'Export to PSD',
    cancelExportPsd: 'Cancel PSD export',
    mergeVisible: 'Merge visible',
    new: 'New layer',
    toggleVisibility: 'Hide all on canvas',
  };
  return {
    useLayerStackActions: (stack: Parameters<typeof getStackActions>[0]) => {
      const ids = getStackActions(stack);
      if (stack === 'raster' && realStackHeader.exporting) {
        ids.splice(2, 0, 'cancelExportPsd');
      }
      return ids.map((id) => ({
        disabled: realStackHeader.exporting && id === 'exportPsd',
        icon: PlusIcon,
        id,
        label: labels[id],
        status: realStackHeader.exporting && id === 'exportPsd' ? 'Writing PSD…' : undefined,
        run: () => {},
      }));
    },
  };
});
const StubStackHeader = ({
  commands,
  focused,
  pinned,
  rowKey,
  stack,
}: {
  commands: { focus(key: string): void; keyDown(key: string, event: React.KeyboardEvent<HTMLElement>): void };
  focused: boolean;
  pinned?: boolean;
  rowKey: string;
  stack: string;
}) => (
  <div
    aria-label={stack}
    aria-level={1}
    data-layer-row-id={pinned ? undefined : rowKey}
    aria-hidden={pinned || undefined}
    data-testid="stack-header"
    role="treeitem"
    tabIndex={pinned ? undefined : focused ? 0 : -1}
    onFocus={() => commands.focus(rowKey)}
    onKeyDown={(event) => commands.keyDown(rowKey, event)}
  >
    {stack}
  </div>
);
vi.mock('./LayerStackHeader', async (importOriginal) => {
  const actual = await importOriginal<typeof LayerStackHeaderModule>();
  return {
    LayerStackHeader: (props: Parameters<typeof actual.LayerStackHeader>[0]) =>
      realStackHeader.on ? <actual.LayerStackHeader {...props} /> : <StubStackHeader {...props} />,
  };
});
const surfaceCommands = vi.hoisted(() => ({ current: null as LayerRowCommands | null }));
vi.mock('./LayerSurfaceHost', () => ({
  LayerSurfaceHost: ({
    commands,
    surface,
  }: {
    commands: LayerRowCommands;
    surface: {
      kind: string;
      id?: string;
      child?: { key: string };
      anchor: { height: number; width: number; x: number; y: number };
    } | null;
  }) => {
    surfaceCommands.current = commands;
    return (
      <output
        data-anchor={surface ? JSON.stringify(surface.anchor) : undefined}
        data-testid="surface"
        style={{ display: 'none' }}
      >
        {surface ? `${surface.kind}:${surface.child?.key ?? surface.id}` : 'none'}
      </output>
    );
  },
}));

const i18n = createInstance();
void i18n.use(initReactI18next).init({
  fallbackLng: 'en',
  initAsync: false,
  lng: 'en',
  resources: {
    en: {
      translation: {
        widgets: {
          layers: {
            actions: {
              collapseGroup: 'Collapse group',
              excludeFromGeneration: 'Exclude from generation',
              excludeGroupFromGeneration: 'Exclude group from generation',
              expandGroup: 'Expand group',
              groupExcluded: 'Left out of generation by a group',
              groupHidden: 'Hidden on the canvas by a group',
              groupLocked: 'Locked by a group',
              hideGroup: 'Hide group',
              hideLayer: 'Hide layer',
              hideOnCanvas: 'Hide on canvas',
              hideModifiers: 'Hide modifiers',
              indent: 'Move into the group above',
              outdent: 'Move out of the group',
              rename: 'Rename',
              reorder: 'Reorder layer',
              select: 'Select {{name}}',
              showModifiers: 'Show modifiers',
              showOnCanvas: 'Show on canvas',
              toggleLock: 'Toggle lock',
              toggleActive: 'Toggle layer active',
              toggleVisibility: 'Toggle visibility',
              useInGeneration: 'Use in generation',
            },
            adjustments: { curves: 'Curves', saturation: 'Saturation' },
            modifiers: {
              brightnessContrast: 'Brightness/Contrast',
              denoise: 'Denoise limit',
              disable: 'Disable',
              duplicateAdjustment: 'Duplicate adjustment',
              enable: 'Enable',
              noise: 'Noise',
              removeAdjustment: 'Remove adjustment',
              removeDenoise: 'Remove denoise limit',
              removeNoise: 'Remove noise',
              parentDisabled: 'The layer is hidden',
              parentExcluded: 'The layer is left out of generation',
              reorderAdjustment: 'Reorder adjustment',
              toggleActive: 'Toggle active',
            },
            regionalGuidance: {
              referenceImage: 'Reference image',
              referenceImages: 'Reference images',
              removeReferenceImage: 'Remove reference image',
            },
            groupSummary_one: '{{count}} layer',
            groupSummary_other: '{{count}} layers',
            groups: { raster: 'Raster Layers' },
            properties: 'Layer properties',
            tree: 'Layer tree',
            types: { paint: 'Paint' },
          },
        },
      },
    },
  },
});

const PROJECT_ID = 'test-project';
const HIDDEN = { display: 'none' } as const;
const paint = (id: string, name = id) => layerContract(id, 'raster', { name });
const manyLayers = (count: number): CanvasNodeContract[] =>
  Array.from({ length: count }, (_, index) => paint(`l${index}`, `Layer ${index}`));

let dispatchExternal: (mutation: CanvasProjectMutation) => void = () => undefined;
const thumbnailRequests = vi.fn();
const exportBakedLayerBlob = vi.fn(() =>
  Promise.resolve({ blob: new Blob(['png'], { type: 'image/png' }), status: 'ok' as const })
);
const refusalChecks = vi.fn();
const committedEdits: PreparedDocumentEdit[] = [];
const revealRequests = vi.fn();

const operationHarness = (() => {
  const listeners = new Set<() => void>();
  let state: CanvasOperationState = { status: 'idle' };
  return {
    getOperationState: () => state,
    set: (next: CanvasOperationState) => {
      state = next;
      listeners.forEach((listener) => listener());
    },
    subscribeOperation: (listener: () => void) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
})();
const withOperations = (engine: LayersTreeEngine): LayersTreeEngine => {
  attachCanvasOperations(engine, operationHarness as never);
  return engine;
};
const adjustmentStatus = createFakeAdjustmentStatus();
let harnessDocument = createEmptyCanvasDocument();
const engine = withOperations({
  document: {
    model: () => {
      const model = createDocumentModel(harnessDocument, { editRevision: 0, projectId: PROJECT_ID });
      return {
        ...model,
        refusalFor: (command: Parameters<typeof model.refusalFor>[0]) => {
          refusalChecks(command);
          return model.refusalFor(command);
        },
      };
    },
  },
  exports: { exportBakedLayerBlob: exportBakedLayerBlob, hasExportableLayerContent: () => false },
  interaction: { get: () => false, ...adjustmentStatus.interaction },
  layers: {
    commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
      committedEdits.push(edit);
      dispatchExternal(edit.forward);
      return { status: 'committed' as const };
    },
    endStructuralPreview: () => undefined,
  },
  previews: { drawLayerThumbnail: () => false, requestLayerThumbnail: thumbnailRequests },
  projectId: PROJECT_ID,
} as unknown as LayersTreeEngine);
const setOperation = (layerId: string | null) =>
  operationHarness.set(
    layerId === null
      ? { status: 'idle' }
      : {
          error: null,
          identity: { kind: 'select-object', layerId, projectId: PROJECT_ID },
          phase: 'ready',
          status: 'active',
        }
  );

const Harness = ({ initialNodes }: { initialNodes: CanvasNodeContract[] }) => {
  const [project, setProject] = useState<Project>(() => {
    const initial = { ...createInitialWorkbenchState().projects[0]!, id: PROJECT_ID };
    return applyCanvasProjectMutation(initial, {
      document: { ...createEmptyCanvasDocument(), selectedLayerId: null, stacks: stacksFrom(initialNodes) },
      type: 'replaceCanvasDocument',
    });
  });
  const document = project.canvas.document;
  useLayoutEffect(() => reconcileLayerPanelStates([project]), [project]);
  const panel = useLayerPanelState(PROJECT_ID, document.selectedLayerId);
  const dispatch = useCallback((mutation: CanvasProjectMutation) => {
    setProject((current) => applyCanvasProjectMutation(current, mutation));
    return true;
  }, []);
  useEffect(() => {
    dispatchExternal = dispatch;
  }, [dispatch]);
  useLayoutEffect(() => {
    harnessDocument = document;
  }, [document]);
  const expanded = useMemo(() => new Set(panel.expandedGroupIds), [panel.expandedGroupIds]);
  const stacks = useMemo(
    () => buildLayerStackRows(document.stacks, expanded, panel.filter),
    [document, expanded, panel.filter]
  );
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 320, width: 480 }}>
      <LayersTree
        degraded={initialNodes.length > LAYER_PANEL_DEGRADE_THRESHOLD}
        dispatch={dispatch}
        document={document}
        editingLocked={false}
        engine={engine}
        panel={panel}
        onRevealProperties={revealRequests}
        projectId={PROJECT_ID}
        stacks={stacks}
      />
      <output data-testid="selected-layer" style={HIDDEN}>
        {document.selectedLayerId ?? 'none'}
      </output>
      <output data-testid="raster-adjustments" style={HIDDEN}>
        {(() => {
          const layer = getDocumentLayer(document, 'r1');
          return layer?.type === 'raster'
            ? (layer.adjustments?.map((entry) => `${entry.id}:${entry.isEnabled ? 'on' : 'off'}`).join(',') ?? 'none')
            : 'none';
        })()}
      </output>
      <output data-testid="adjustment-owners" style={HIDDEN}>
        {getDocumentIndex(document)
          .nodes.flatMap(({ node }) =>
            'adjustments' in node && node.adjustments?.length
              ? [`${node.id}[${node.adjustments.map((entry) => entry.id).join(',')}]`]
              : []
          )
          .join(' ') || 'none'}
      </output>
      <output data-testid="mask-modifiers" style={HIDDEN}>
        {(() => {
          const layer = getDocumentLayer(document, 'mask');
          return layer?.type === 'inpaint_mask'
            ? [
                layer.noise ? `noise:${layer.noise.isEnabled ? 'on' : 'off'}:${layer.noise.level}` : null,
                layer.denoise ? `denoise:${layer.denoise.isEnabled ? 'on' : 'off'}:${layer.denoise.limit}` : null,
              ]
                .filter(Boolean)
                .join(',') || 'empty'
            : 'none';
        })()}
      </output>
      <output data-testid="regional-refs" style={HIDDEN}>
        {['rg', 'rg2']
          .map((id) => {
            const layer = getDocumentLayer(document, id);
            return layer?.type === 'regional_guidance'
              ? `${id}[${layer.referenceImages.map((ref) => `${ref.id}:${ref.isEnabled ? 'on' : 'off'}`).join(',')}]`
              : null;
          })
          .filter(Boolean)
          .join(' ') || 'none'}
      </output>
      <output data-testid="selected-layers" style={HIDDEN}>
        {panel.selectedIds.join(',') || 'none'}
      </output>
      <output data-testid="layer-order" style={HIDDEN}>
        {getDocumentLeaves(document)
          .map((layer) => layer.id)
          .join(',')}
      </output>
    </div>
  );
};

let host: HTMLDivElement | null = null;
let root: Root | null = null;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const renderTree = async (initialNodes: CanvasNodeContract[]) => {
  host = document.createElement('div');
  document.body.append(host);
  root = createRoot(host);
  await act(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>
          <Harness initialNodes={initialNodes} />
        </ChakraProvider>
      </I18nextProvider>
    );
  });
  // The virtualizer sizes its window from a ResizeObserver report, which lands after the first paint.
  await settle();
};

const settle = () =>
  act(
    () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
      })
  );

const treeitems = (): HTMLElement[] => Array.from(host!.querySelectorAll<HTMLElement>('[role="treeitem"]'));
const treeitem = (name: string): HTMLElement =>
  host!.querySelector<HTMLElement>(`[role="treeitem"][aria-label="${name}"]`)!;
const tabStops = (): HTMLElement[] =>
  Array.from(
    host!.querySelectorAll<HTMLElement>('[role="tree"] button, [role="tree"] input, [role="tree"] [tabindex]')
  ).filter((element) => element.tabIndex >= 0);
const output = (id: string): string => host!.querySelector<HTMLOutputElement>(`[data-testid="${id}"]`)!.value;
const pointer = (type: string, target: EventTarget, clientX: number, clientY: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX, clientY, isPrimary: true, pointerId: 1 })
  );
};
const centre = (element: Element) => {
  const rect = element.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
};

beforeEach(() => {
  clearLayerPanelStates();
  clearLayerChildSelection();
  resetLayerRowCommits();
  thumbnailRequests.mockClear();
  refusalChecks.mockClear();
  committedEdits.length = 0;
  revealRequests.mockClear();
  clearLayerPropertiesRequest();
  setOperation(null);
});

afterEach(async () => {
  await act(() => root?.unmount());
  realStackHeader.on = false;
  realStackHeader.exporting = false;
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

describe('LayersTree virtualization', () => {
  it('mounts only the rows near the viewport of a 2,000-layer document and keeps the tree height exact', async () => {
    await renderTree(manyLayers(2000));
    const visible = Math.ceil(320 / LAYER_ROW_HEIGHT_PX);
    expect(treeitems().length).toBeLessThanOrEqual(visible + 12 + 2);
    expect(treeitems().length).toBeGreaterThanOrEqual(visible);
    expect(thumbnailRequests.mock.calls.length).toBeLessThanOrEqual(treeitems().length);
    const tree = host!.querySelector<HTMLElement>('[role="tree"]')!;
    expect(tree.getAttribute('aria-multiselectable')).toBe('true');
    expect(getComputedStyle(tree).height).toBe(`${2000 * LAYER_ROW_HEIGHT_PX + 28}px`);
  });

  it('commits only the rows a selection change touches, and nothing on scroll', async () => {
    await renderTree(manyLayers(2000));
    await act(() => userEvent.click(treeitem('Layer 1')));
    resetLayerRowCommits();
    thumbnailRequests.mockClear();
    await act(() => userEvent.click(treeitem('Layer 3')));
    expect(output('selected-layer')).toBe('l3');
    expect(Object.keys(getLayerRowCommits()).sort()).toEqual(['l1', 'l3']);
    expect(thumbnailRequests).not.toHaveBeenCalled();

    resetLayerRowCommits();
    const scroller = host!.querySelector<HTMLElement>('[role="tree"]')!.closest<HTMLElement>('[data-part="viewport"]')!;
    await act(() => {
      scroller.scrollTop = 7;
      scroller.dispatchEvent(new Event('scroll'));
    });
    await settle();
    expect(getLayerRowCommits()).toEqual({});
  });
});

describe('LayersTree keyboard and accessibility', () => {
  const nested = (): CanvasNodeContract[] => [
    paint('top', 'Top'),
    groupContract(
      'g',
      [paint('inner', 'Inner'), groupContract('h', [paint('deep', 'Deep')], { name: 'Inner group' })],
      {
        isLocked: true,
        name: 'Group',
      }
    ),
    paint('bottom', 'Bottom'),
  ];

  it('exposes one tab stop across every control and full tree semantics', async () => {
    await renderTree(nested());
    expect(tabStops()).toHaveLength(1);
    expect(host!.querySelectorAll('[role="tree"] > *')).toHaveLength(
      host!.querySelectorAll('[role="tree"] > [role="presentation"]').length
    );
    const group = treeitem('Group');
    expect(group).toHaveAttribute('aria-level', '2');
    expect(group).toHaveAttribute('aria-posinset', '2');
    expect(group).toHaveAttribute('aria-setsize', '3');
    expect(group).toHaveAttribute('aria-expanded', 'false');
    expect(group).toHaveAttribute('aria-selected', 'false');
  });

  it('walks rows with the arrow keys, opens and enters groups, climbs to the parent, and jumps with End', async () => {
    await renderTree(nested());
    treeitem('Top').focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    expect(document.activeElement).toBe(treeitem('Group'));
    expect(treeitem('Group').tabIndex).toBe(0);
    expect(treeitem('Top').tabIndex).toBe(-1);
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(treeitem('Group')).toHaveAttribute('aria-expanded', 'true');
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(document.activeElement).toBe(treeitem('Inner'));
    expect(treeitem('Inner')).toHaveAttribute('aria-level', '3');
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(document.activeElement).toBe(treeitem('Group'));
    await act(() => userEvent.keyboard('{End}'));
    expect(document.activeElement).toBe(treeitem('Bottom'));
    await act(() => userEvent.keyboard('{Home}'));
    expect(document.activeElement).toBe(treeitem('raster'));
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(host!.querySelectorAll('[role="treeitem"]')).toHaveLength(1);
    await act(() => userEvent.keyboard('{ArrowRight}{ArrowRight}'));
    expect(document.activeElement).toBe(treeitem('Top'));
    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
    expect(output('surface')).toBe('menu:top');
    const row = treeitem('Top').closest('[data-layer-row-id]')!.getBoundingClientRect();
    expect(JSON.parse(host!.querySelector<HTMLElement>('[data-testid="surface"]')!.dataset.anchor!)).toEqual({
      height: 1,
      width: 1,
      x: row.left,
      y: row.bottom,
    });
  });

  it('keeps the focused row mounted and focused while scrolled far away', async () => {
    await renderTree(manyLayers(2000));
    treeitem('Layer 0').focus();
    await act(() => userEvent.keyboard('{End}'));
    await settle();
    expect(document.activeElement).toBe(treeitem('Layer 1999'));
    const scroller = host!.querySelector<HTMLElement>('[role="tree"]')!.closest<HTMLElement>('[data-part="viewport"]')!;
    await act(() => {
      scroller.scrollTop = 0;
      scroller.dispatchEvent(new Event('scroll'));
    });
    await settle();
    expect(treeitem('Layer 1999')).not.toBeNull();
    expect(treeitem('Layer 1999').tabIndex).toBe(0);
  });
});

describe('LayersTree selection, surfaces and structure', () => {
  const trio = (): CanvasNodeContract[] => [
    paint('first', 'First'),
    paint('second', 'Second'),
    paint('third', 'Third'),
  ];

  it('selects with click, toggles with Ctrl, ranges with Shift', async () => {
    await renderTree(trio());
    await act(() => userEvent.click(treeitem('First')));
    expect(output('selected-layer')).toBe('first');
    await act(() => treeitem('Third').dispatchEvent(new MouseEvent('click', { bubbles: true, ctrlKey: true })));
    expect(output('selected-layers')).toBe('first,third');
    expect(treeitem('Third')).toHaveAttribute('aria-current', 'true');
    await act(() => treeitem('First').dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true })));
    expect(output('selected-layers')).toBe('first,second,third');
  });

  it('deletes the focused row with Delete and the selection with Backspace, keeping focus in the tree', async () => {
    await renderTree(trio());
    treeitem('Second').focus();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(output('layer-order')).toBe('first,third');
    expect(document.activeElement).toBe(treeitem('Third'));

    treeitem('Third').focus();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(output('layer-order')).toBe('first');
    expect(document.activeElement).toBe(treeitem('First'));

    await act(() => userEvent.click(treeitem('First')));
    await act(() => userEvent.keyboard('{Backspace}'));
    expect(output('layer-order')).toBe('');
  });

  describe('overlay rows keep generation and canvas visibility apart', () => {
    const leaf = (id: string) => compileDocumentLeaves(harnessDocument).find((entry) => entry.id === id)!;
    const control = (id: string, label: string) =>
      host!.querySelector<HTMLButtonElement>(`[data-layer-row-id="${id}"] button[aria-label="${label}"]`)!;
    const dotOf = (id: string) => control(id, 'Use in generation');
    const eyeOf = (id: string) => control(id, 'Show on canvas');
    const state = (id: string) => {
      const { contributionEnabled, documentHidden, layer } = leaf(id);
      return { contributionEnabled, documentHidden, isEnabled: layer.isEnabled, isHidden: isNodeHidden(layer) };
    };

    it('drives an overlay group from its own dot and eye, and shows its members gated by it', async () => {
      await renderTree([
        groupContract('rg', [layerContract('member', 'regional_guidance', { name: 'Member' })], { name: 'Regions' }),
        layerContract('loose', 'regional_guidance', { isEnabled: false, name: 'Loose' }),
      ]);
      await act(() => userEvent.click(treeitem('Regions').querySelector('button[aria-label="Expand group"]')!));
      const dotFill = (id: string) => getComputedStyle(dotOf(id), '::before').backgroundColor;
      const dotRing = (id: string) => getComputedStyle(dotOf(id), '::before').borderColor;
      const ringStyle = (id: string) => getComputedStyle(dotOf(id), '::before').borderTopStyle;
      const description = (id: string) => {
        const describedBy = dotOf(id).getAttribute('aria-describedby');
        return describedBy ? document.getElementById(describedBy)?.textContent : null;
      };
      expect(description('member')).toBeNull();
      const accent = dotFill('member');
      expect(accent).not.toBe('rgba(0, 0, 0, 0)');

      await act(() => userEvent.click(dotOf('rg')));
      expect(getDocumentIndex(harnessDocument).byId.get('rg')!.node.isEnabled).toBe(false);
      expect(state('member')).toMatchObject({ contributionEnabled: false, isEnabled: true });
      expect(dotOf('member')).toHaveAttribute('aria-pressed', 'true');
      expect(description('member')).toBe('Left out of generation by a group');
      await expect.poll(() => dotFill('member')).toBe('rgba(0, 0, 0, 0)');
      expect(dotRing('member')).toBe(accent);
      expect(ringStyle('member')).toBe('dashed');
      expect(dotFill('loose')).toBe('rgba(0, 0, 0, 0)');
      expect(ringStyle('loose')).toBe('solid');
      expect(description('loose')).toBeNull();
      await act(() => userEvent.click(dotOf('rg')));
      expect(ringStyle('member')).toBe('solid');
      expect(description('member')).toBeNull();

      await act(() => userEvent.click(eyeOf('rg')));
      expect(isNodeHidden(getDocumentIndex(harnessDocument).byId.get('rg')!.node)).toBe(true);
      expect(state('member')).toEqual({
        contributionEnabled: true,
        documentHidden: true,
        isEnabled: true,
        isHidden: false,
      });
      expect(eyeOf('member')).toHaveAttribute('aria-pressed', 'false');
      expect(eyeOf('member').disabled).toBe(true);
      expect(dotOf('member').disabled).toBe(false);
    });
  });

  it('reveals a cycled layer through collapsed ancestors and a filter without opening properties', async () => {
    await renderTree([
      ...manyLayers(60),
      groupContract('outer', [groupContract('inner', [layerContract('target', 'raster', { name: 'Target' })])]),
    ]);
    await act(() => dispatchExternal({ id: 'l59', type: 'setCanvasSelectedLayer' }));
    await act(() => {
      setLayerPanelFilter(PROJECT_ID, 'l59', 'Layer 59');
      toggleLayerStackCollapsed(PROJECT_ID, 'l59', 'raster');
    });
    const ctx: CanvasHotkeyContext = {
      document: harnessDocument,
      engine: engine as CanvasEngine,
      dispatch: (mutation) => {
        dispatchExternal(mutation);
        return true;
      },
      selectedLayerIds: ['l59'],
      hasMultipleStagingSlots: false,
      copySelection: vi.fn(),
      cropAllLayers: vi.fn(),
      swapActiveColors: vi.fn(),
      resetActiveColors: vi.fn(),
      pasteFromClipboard: vi.fn(),
      notifyLayerDuplicateFailed: vi.fn(),
      reportStructuralCommit: vi.fn(),
      reportPreparedCommit: vi.fn(),
      reportLayerOperation: vi.fn(),
      reportMergeDownRefusal: vi.fn(),
      reportTransformRefusal: vi.fn(),
      t: (key) => key,
    };
    await act(() => executeCanvasHotkeyCommand('canvas.nextEntity', ctx));
    await settle();
    expect(output('selected-layer')).toBe('target');
    expect(readLayerPanelState(PROJECT_ID, 'target')).toMatchObject({
      expandedGroupIds: ['outer', 'inner'],
      collapsedStacks: [],
      filter: '',
    });
    const row = treeitem('Target');
    const scroller = host!.querySelector<HTMLElement>('[role="tree"]')!.closest<HTMLElement>('[data-part="viewport"]')!;
    expect(scroller.scrollTop).toBeGreaterThan(0);
    expect(row.getBoundingClientRect().top).toBeGreaterThanOrEqual(scroller.getBoundingClientRect().top);
    expect(row.getBoundingClientRect().bottom).toBeLessThanOrEqual(scroller.getBoundingClientRect().bottom);
    expect(output('surface')).toBe('none');
    expect(revealRequests).not.toHaveBeenCalled();
  });

  it('reorders with a pointer drag and reparents from the keyboard', async () => {
    await renderTree([
      paint('first', 'First'),
      groupContract('g', [paint('inner', 'Inner')], { name: 'Group' }),
      paint('third', 'Third'),
    ]);
    const first = host!.querySelector<HTMLElement>('[data-layer-row-id="first"]')!;
    const third = host!.querySelector<HTMLElement>('[data-layer-row-id="third"]')!;
    const start = centre(first);
    const end = centre(third);
    await act(() => pointer('pointerdown', first, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, end.x, end.y + 12));
    await act(() => pointer('pointerup', document, end.x, end.y + 12));
    expect(output('layer-order')).toBe('inner,third,first');

    treeitem('Third').focus();
    await act(() => userEvent.keyboard('{Alt>}{ArrowRight}{/Alt}'));
    expect(output('layer-order')).toBe('inner,third,first');
    expect(readLayerPanelState(PROJECT_ID, null).expandedGroupIds).toContain('g');
    expect(treeitem('Third')).toHaveAttribute('aria-level', '3');
  });
});

describe('LayersTree projected child rows', () => {
  const referenceImage = (id: string, isEnabled = true) => ({
    config: {
      beginEndStepPct: [0, 1] as [number, number],
      clipVisionModel: 'ViT-H' as const,
      image: null,
      method: 'full' as const,
      model: null,
      type: 'ip_adapter' as const,
      weight: 1,
    },
    id,
    isEnabled,
  });
  const regionWithRefs = (): CanvasNodeContract[] => [
    layerContract('rg', 'regional_guidance', {
      name: 'Region',
      referenceImages: [referenceImage('ref1'), referenceImage('ref2')],
    }),
    paint('r1', 'Raster'),
  ];

  it('renders one row per reference image with tree semantics, and the chevron hides them', async () => {
    await renderTree(regionWithRefs());
    const ref1 = treeitem('Reference image 1');
    expect(ref1).toHaveAttribute('aria-level', '3');
    expect(ref1).toHaveAttribute('aria-posinset', '1');
    expect(ref1).toHaveAttribute('aria-setsize', '2');
    expect(treeitem('Region')).toHaveAttribute('aria-expanded', 'true');
    await act(() => userEvent.click(host!.querySelector('button[aria-label="Hide modifiers"]')!));
    expect(host!.querySelector('[role="treeitem"][aria-label="Reference image 1"]')).toBeNull();
    expect(treeitem('Region')).toHaveAttribute('aria-expanded', 'false');
    await act(() => userEvent.click(host!.querySelector('button[aria-label="Show modifiers"]')!));
    expect(treeitem('Reference image 2')).not.toBeNull();
  });

  it("builds a child row's edit from the engine's document, not the rendered one", async () => {
    await renderTree([
      layerContract('mask', 'inpaint_mask', { name: 'Mask', noise: { isEnabled: true, level: 0.25 } }),
    ]);
    harnessDocument = {
      ...harnessDocument,
      stacks: stacksFrom([
        layerContract('mask', 'inpaint_mask', { name: 'Mask', noise: { isEnabled: true, level: 0.5 } }),
      ]),
    };
    const eye = treeitem('Noise').querySelector<HTMLButtonElement>('button[aria-label="Toggle active"]')!;
    await act(() => userEvent.click(eye));

    expect(committedEdits.at(-1)!.forward).toMatchObject({ config: { noise: { isEnabled: false, level: 0.5 } } });
  });

  describe('moving adjustments between layers', () => {
    const adjustedLayers = (overrides: { target?: object } = {}): CanvasNodeContract[] => [
      layerContract('r1', 'raster', {
        adjustments: [
          { brightness: 0.2, contrast: 0, id: 'a1', isEnabled: true, type: 'brightness-contrast' },
          { id: 'a2', isEnabled: true, saturation: -0.4, type: 'hsl' },
        ],
        name: 'Source',
      }),
      layerContract('r2', 'raster', {
        adjustments: [{ curves: {}, id: 'b1', isEnabled: true, type: 'curves' }],
        name: 'Target',
        ...overrides.target,
      }),
      groupContract('g1', [paint('r3', 'Member')], { name: 'Group' }),
      layerContract('c1', 'control', { name: 'Control' }),
    ];
    const row = (key: string) => host!.querySelector<HTMLElement>(`[data-layer-row-id="${key}"]`)!;
    const drag = async (from: HTMLElement, to: HTMLElement, dy = 0) => {
      const start = centre(from);
      const end = centre(to);
      await act(() => pointer('pointerdown', from, start.x, start.y));
      await act(() => pointer('pointermove', document, start.x + 8, start.y));
      await act(() => pointer('pointermove', document, end.x, end.y + dy));
      await act(() => pointer('pointerup', document, end.x, end.y + dy));
      await settle();
    };

    it('lands among another layer’s adjustments as one undoable edit, and the selection follows', async () => {
      await renderTree(adjustedLayers());
      await drag(row('child:r1:a2'), row('child:r2:b1'), -6);
      expect(output('adjustment-owners')).toBe('r1[a1] r2[a2,b1]');
      expect(output('selected-layer')).toBe('r2');
      expect(getLayerChildSelection()).toMatchObject({ itemId: 'a2', layerId: 'r2' });
      expect(document.activeElement).toBe(row('child:r2:a2'));

      expect(committedEdits).toHaveLength(1);
      await act(() => dispatchExternal(committedEdits[0]!.inverse));
      expect(output('adjustment-owners')).toBe('r1[a1,a2] r2[b1]');
    });

    it('refuses control layers and locked raster layers, leaving every owner intact', async () => {
      await renderTree(adjustedLayers({ target: { isLocked: true } }));
      await drag(row('child:r1:a1'), row('c1'));
      await drag(row('child:r1:a1'), row('r2'));
      await drag(row('child:r1:a1'), row('child:r2:b1'), -6);
      expect(output('adjustment-owners')).toBe('r1[a1,a2] r2[b1]');
      expect(committedEdits).toHaveLength(0);
    });
  });
});

describe('LayersTree list changes', () => {
  it('survives the list shrinking under the virtual window and repairs focus onto the tab stop', async () => {
    await renderTree(manyLayers(50));
    treeitem('Layer 0').focus();
    await act(() => userEvent.keyboard('{End}'));
    await settle();
    expect(document.activeElement).toBe(treeitem('Layer 49'));
    await act(() => {
      host!.querySelector<HTMLElement>('[role="tree"]')!.dispatchEvent(new Event('scroll', { bubbles: true }));
    });
    await act(() => setLayerPanelFilter(PROJECT_ID, readLayerPanelState(PROJECT_ID, null).primaryId, 'Layer 1'));
    await settle();
    expect(treeitems().length).toBeGreaterThan(0);
    expect(treeitems().filter((item) => item.tabIndex === 0)).toHaveLength(1);
    expect(host!.contains(document.activeElement)).toBe(true);
    await act(() => setLayerPanelFilter(PROJECT_ID, readLayerPanelState(PROJECT_ID, null).primaryId, ''));
    await settle();
    expect(treeitems().length).toBeGreaterThan(8);
  });

  it('keeps the focused node and its rename draft when rows above it come and go', async () => {
    await renderTree([layerContract('c1', 'control', { name: 'Control' }), ...manyLayers(30)]);
    const row = treeitem('Layer 3');
    row.focus();
    await act(() => userEvent.keyboard('{F2}'));
    const input = host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    await act(() => userEvent.keyboard('abc'));
    await act(() => toggleLayerStackCollapsed(PROJECT_ID, readLayerPanelState(PROJECT_ID, null).primaryId, 'control'));
    await settle();
    expect(host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')).toBe(input);
    expect(input.value).toBe('abc');
    await act(() => userEvent.keyboard('{Escape}'));
    expect(treeitem('Layer 3')).toBe(row);
    expect(document.activeElement).toBe(row);
    await act(() => toggleLayerStackCollapsed(PROJECT_ID, readLayerPanelState(PROJECT_ID, null).primaryId, 'control'));
    await settle();
    expect(document.activeElement).toBe(treeitem('Layer 3'));
    expect(treeitem('Control')).not.toBeNull();
  });
});
