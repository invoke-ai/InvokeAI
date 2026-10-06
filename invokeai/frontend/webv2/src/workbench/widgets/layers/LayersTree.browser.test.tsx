/* oxlint-disable react-perf/jsx-no-new-function-as-prop */
import type { CanvasNodeContract, PreparedDocumentEdit } from '@workbench/canvas-engine/api';
import type { CanvasOperationState } from '@workbench/canvas-operations/api';
import type { CanvasProjectMutation } from '@workbench/canvasProjectMutations';
import type { Project } from '@workbench/projectContracts';

import { ChakraProvider } from '@chakra-ui/react';
import { system } from '@theme/system';
import {
  createDocumentModel,
  getDocumentIndex,
  getDocumentLayer,
  getDocumentLeaves,
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
import { createInitialWorkbenchState } from '@workbench/workbenchState';
import { createInstance } from 'i18next';
import { act, useCallback, useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import type { ProjectedChildRow } from './layerChildRows';
import type { LayerRowCommands } from './layerRowCommands';

import { clearLayerChildSelection, getLayerChildSelection } from './layerChildSelection';
import { getLayerRowCommits, resetLayerRowCommits } from './layerPanelDiagnostics';
import { LAYER_PANEL_DEGRADE_THRESHOLD, LAYER_ROW_HEIGHT_PX } from './layerPanelRows';
import { clearLayerPropertiesRequest, requestLayerProperties } from './layerPropertiesRequestStore';
import { LayersTree, type LayersTreeEngine } from './LayersTree';
import { buildLayerStackRows } from './layerTreeRows';

vi.mock('./ControlLayerWarningIcon', () => ({ ControlLayerWarningIcon: () => null }));
vi.mock('./LayerThumbnail', () => ({ LayerThumbnail: () => <span data-testid="thumbnail" /> }));
vi.mock('./LayerStackHeader', () => ({
  LayerStackHeader: ({
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
  ),
}));
/** The command handle the tree last gave its surface host, which menus drive. */
const surfaceCommands = vi.hoisted(() => ({ current: null as LayerRowCommands | null }));
vi.mock('./LayerSurfaceHost', () => ({
  LayerSurfaceHost: ({
    commands,
    surface,
  }: {
    commands: LayerRowCommands;
    surface: { kind: string; id?: string; child?: { key: string } } | null;
  }) => {
    surfaceCommands.current = commands;
    return (
      // Hidden like the harness outputs so the tree's scroll geometry doesn't depend on this stand-in's line height.
      <output data-testid="surface" style={{ display: 'none' }}>
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
              expandGroup: 'Expand group',
              groupLocked: 'Locked by a group',
              hideModifiers: 'Hide modifiers',
              indent: 'Move into the group above',
              outdent: 'Move out of the group',
              rename: 'Rename',
              reorder: 'Reorder layer',
              select: 'Select {{name}}',
              showModifiers: 'Show modifiers',
              toggleLock: 'Toggle lock',
              toggleActive: 'Toggle layer active',
              toggleVisibility: 'Toggle visibility',
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

/** The one operation-state seam the tree reads; tests publish through `setOperation`. */
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
// The real engine handle is stable across document changes; this one reads the harness's latest document.
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
  interaction: { get: () => false },
  layers: {
    commitPrepared: (_label: string, edit: PreparedDocumentEdit) => {
      committedEdits.push(edit);
      dispatchExternal(edit.forward);
      return { status: 'committed' as const };
    },
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
  // Production reconciles the panel against every document change before the panel reads it.
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
/** Every element inside the tree a Tab press could land on. */
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
    // Thumbnails are asked for once per mounted leaf and never again for the rest of the list.
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

  it('degrades a document above the threshold: no thumbnails, no drag, every command intact', async () => {
    await renderTree(manyLayers(LAYER_PANEL_DEGRADE_THRESHOLD + 1));
    expect(host!.querySelectorAll('[data-testid="thumbnail"]')).toHaveLength(0);
    expect(thumbnailRequests).not.toHaveBeenCalled();
    const first = host!.querySelector<HTMLElement>('[data-layer-row-id="l0"]')!;
    const second = host!.querySelector<HTMLElement>('[data-layer-row-id="l1"]')!;
    const start = centre(first);
    const end = centre(second);
    await act(() => pointer('pointerdown', first, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, end.x, end.y + 12));
    await act(() => pointer('pointerup', document, end.x, end.y + 12));
    expect(output('layer-order').startsWith('l0,l1,l2')).toBe(true);
    treeitem('Layer 0').focus();
    await act(() => userEvent.keyboard('{Alt>}{ArrowDown}{/Alt}'));
    expect(output('layer-order').startsWith('l1,l0,l2')).toBe(true);
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

  it('copies the focused layer to the clipboard on ctrl+c', async () => {
    await renderTree(nested());
    const write = vi.spyOn(navigator.clipboard, 'write').mockResolvedValue();
    try {
      treeitem('Top').focus();
      await act(() => userEvent.keyboard('{Control>}c{/Control}'));
      await expect.poll(() => write.mock.calls.length).toBe(1);
      expect(exportBakedLayerBlob).toHaveBeenCalledWith('top', { includeDisabled: true });
    } finally {
      write.mockRestore();
    }
  });

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
  });

  it('frames the keyboard-focused row surface with its ring, not the gap below it', async () => {
    await renderTree(nested());
    treeitem('Top').focus();
    await act(() => userEvent.keyboard('{ArrowDown}'));
    const row = treeitem('Group');
    expect(document.activeElement).toBe(row);

    const ring = getComputedStyle(row, '::after');
    const rowRect = row.getBoundingClientRect();
    const surface = [...row.children].find((child) => child.textContent?.includes('Group'))!.getBoundingClientRect();
    // The ring and the surface share their box, so the row's content sits centred inside the ring.
    expect(rowRect.top + Number.parseFloat(ring.top)).toBeCloseTo(surface.top, 0);
    expect(rowRect.bottom - Number.parseFloat(ring.bottom)).toBeCloseTo(surface.bottom, 0);
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

  it('reports the inherited lock on rows under a locked group', async () => {
    await renderTree(nested());
    await act(() => userEvent.click(host!.querySelector<HTMLButtonElement>('button[aria-label="Expand group"]')!));
    const innerRow = host!.querySelector<HTMLElement>('[data-layer-row-id="inner"]')!;
    const lock = innerRow.querySelector<HTMLButtonElement>('button[aria-label="Toggle lock"]')!;
    expect(lock.disabled).toBe(true);
    expect(lock.querySelector('svg.lucide-lock')).not.toBeNull();
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

    // The last row falls back to the row above it.
    treeitem('Third').focus();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(output('layer-order')).toBe('first');
    expect(document.activeElement).toBe(treeitem('First'));

    await act(() => userEvent.click(treeitem('First')));
    await act(() => userEvent.keyboard('{Backspace}'));
    expect(output('layer-order')).toBe('');
  });

  it('keeps the visibility dot isolated from row selection', async () => {
    await renderTree(trio());
    const dot = host!.querySelector<HTMLButtonElement>(
      '[data-layer-row-id="first"] button[aria-label="Toggle layer active"]'
    )!;
    await act(() => userEvent.click(dot));
    expect(dot).toHaveAttribute('aria-pressed', 'false');
    expect(output('selected-layer')).toBe('none');
  });

  it('routes the row menu and right-click to the panel surface host by id', async () => {
    await renderTree(trio());
    await act(() =>
      treeitem('Second').dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 20, clientY: 20 })
      )
    );
    expect(output('surface')).toBe('menu:second');
    await act(() =>
      treeitem('Third').dispatchEvent(
        new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 20, clientY: 20 })
      )
    );
    expect(output('surface')).toBe('menu:third');
    expect(output('selected-layer')).toBe('third');
  });

  it('reveals a primary that changed outside the panel once, even inside a collapsed stack', async () => {
    await renderTree([layerContract('c1', 'control', { name: 'Control' }), ...manyLayers(2000)]);
    const scroller = host!.querySelector<HTMLElement>('[role="tree"]')!.closest<HTMLElement>('[data-part="viewport"]')!;
    expect(scroller.scrollTop).toBe(0);
    await act(() => dispatchExternal({ id: 'l1500', type: 'setCanvasSelectedLayer' }));
    await settle();
    expect(scroller.scrollTop).toBeGreaterThan(1000 * LAYER_ROW_HEIGHT_PX);
    expect(treeitem('Layer 1500')).not.toBeNull();

    // Later list changes leave the scroll position alone.
    await act(() => toggleLayerStackCollapsed(PROJECT_ID, 'l1500', 'control'));
    scroller.scrollTop = 0;
    await act(() => toggleLayerStackCollapsed(PROJECT_ID, 'l1500', 'control'));
    await settle();
    expect(scroller.scrollTop).toBe(0);

    // A panel selection reveals nothing, but a later external change back to it does.
    scroller.scrollTop = 1500 * LAYER_ROW_HEIGHT_PX;
    await settle();
    await act(() => userEvent.click(treeitem('Layer 1501')));
    const outside = document.createElement('input');
    document.body.append(outside);
    outside.focus();
    scroller.scrollTop = 0;
    await act(() => dispatchExternal({ id: 'l1500', type: 'setCanvasSelectedLayer' }));
    await settle();
    scroller.scrollTop = 0;
    await act(() => dispatchExternal({ id: 'l1501', type: 'setCanvasSelectedLayer' }));
    await settle();
    expect(scroller.scrollTop).toBeGreaterThan(1000 * LAYER_ROW_HEIGHT_PX);
    outside.remove();

    // A primary inside a collapsed stack opens the stack and lands in view.
    await act(() => toggleLayerStackCollapsed(PROJECT_ID, 'l1500', 'control'));
    await act(() => dispatchExternal({ id: 'c1', type: 'setCanvasSelectedLayer' }));
    await settle();
    expect(readLayerPanelState(PROJECT_ID, 'c1').collapsedStacks).toEqual([]);
    expect(treeitem('Control')).not.toBeNull();
    expect(scroller.scrollTop).toBe(0);
  });

  it('validates a semantic drop command only once across equivalent adjacent hit regions', async () => {
    await renderTree(trio());
    const first = host!.querySelector<HTMLElement>('[data-layer-row-id="first"]')!;
    const second = host!.querySelector<HTMLElement>('[data-layer-row-id="second"]')!;
    const third = host!.querySelector<HTMLElement>('[data-layer-row-id="third"]')!;
    const start = centre(first);
    const secondRect = second.getBoundingClientRect();
    const thirdRect = third.getBoundingClientRect();
    await act(() => pointer('pointerdown', first, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, secondRect.bottom - 2));
    await settle();
    expect(refusalChecks).toHaveBeenCalled();
    const belowSecond = JSON.stringify(refusalChecks.mock.calls.at(-1)?.[0]);

    refusalChecks.mockClear();
    await act(() => pointer('pointermove', document, start.x + 8, thirdRect.top + 2));
    await settle();
    expect(refusalChecks).not.toHaveBeenCalled();
    await act(() => pointer('pointerup', document, start.x + 8, thirdRect.top + 2));

    // Below the second row and above the third row describe the same insertion gap.
    expect(belowSecond).toBe(JSON.stringify({ beforeId: 'third', ids: ['first'], parentId: null, type: 'reparent' }));
  });

  it('scrolls the list while a drag rests in the edge band and asks the model once per target', async () => {
    await renderTree(manyLayers(200));
    const scroller = host!.querySelector<HTMLElement>('[role="tree"]')!.closest<HTMLElement>('[data-part="viewport"]')!;
    const first = host!.querySelector<HTMLElement>('[data-layer-row-id="l0"]')!;
    const start = centre(first);
    const rect = scroller.getBoundingClientRect();
    await act(() => pointer('pointerdown', first, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, start.x, rect.bottom - 4));
    await new Promise<void>((resolve) => {
      setTimeout(resolve, 120);
    });
    expect(scroller.scrollTop).toBeGreaterThan(0);
    await act(() => pointer('pointermove', document, start.x, rect.top + rect.height / 2));
    // Wait for ten unchanged animation frames, not timers, because both dnd movement delivery and autoscroll are
    // rAF-driven.
    const nextFrame = () =>
      new Promise<void>((resolve) => {
        requestAnimationFrame(() => resolve());
      });
    let settled = scroller.scrollTop;
    let stillFrames = 0;
    for (let attempt = 0; attempt < 600 && stillFrames < 10; attempt += 1) {
      await nextFrame();
      const next = scroller.scrollTop;
      if (next === settled) {
        stillFrames += 1;
      } else {
        settled = next;
        stillFrames = 0;
      }
    }
    expect(stillFrames).toBe(10);
    expect(scroller.scrollTop).toBe(settled);
    // "Once per target": a starved rAF can flap the target back to a row
    // already checked, so assert no re-check for the target still under the
    // pointer — the memoized check's contract — not global uniqueness.
    const checked = refusalChecks.mock.calls.map(([command]) => JSON.stringify(command));
    await act(() => pointer('pointerup', document, start.x, rect.top + rect.height / 2));
    expect(checked.filter((command, index) => command === checked[index - 1])).toEqual([]);
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

  it("reveals Properties for a running operation's layer but drops requests for any other layer", async () => {
    await renderTree(trio());
    // An operation started from a menu and its reveal request land in the same batch.
    await act(() => {
      setOperation('second');
      requestLayerProperties('second');
    });
    await settle();
    expect(revealRequests).toHaveBeenCalledWith('second');

    revealRequests.mockClear();
    await act(() => requestLayerProperties('third'));
    await settle();
    expect(revealRequests).not.toHaveBeenCalled();
    expect(output('selected-layer')).not.toBe('third');
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

  it('toggles the item from the dot and removes it with Delete, leaving siblings alone', async () => {
    await renderTree(regionWithRefs());
    const dot = treeitem('Reference image 1').querySelector<HTMLButtonElement>('button[aria-label="Toggle active"]')!;
    await act(() => userEvent.click(dot));
    expect(output('regional-refs')).toBe('rg[ref1:off,ref2:on]');
    expect(output('selected-layer')).toBe('none');
    treeitem('Reference image 2').focus();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(output('regional-refs')).toBe('rg[ref1:off]');
  });

  it('selecting a child selects its owner, records the sub-selection, and reveals Properties', async () => {
    await renderTree(regionWithRefs());
    await act(() => userEvent.click(treeitem('Reference image 2')));
    expect(output('selected-layer')).toBe('rg');
    expect(getLayerChildSelection()).toMatchObject({ itemId: 'ref2', layerId: 'rg' });
    expect(revealRequests).toHaveBeenCalledWith('rg');
    // Selecting a layer row clears the sub-selection again.
    await act(() => userEvent.click(treeitem('Raster')));
    expect(getLayerChildSelection()).toBeNull();
  });

  it('projects mask noise and denoise rows whose dots and Delete edit the modifiers', async () => {
    await renderTree([
      layerContract('mask', 'inpaint_mask', {
        denoise: { isEnabled: true, limit: 0.8 },
        name: 'Mask',
        noise: { isEnabled: true, level: 0.25 },
      }),
    ]);
    expect(treeitem('Noise')).toHaveAttribute('aria-level', '3');
    expect(treeitem('Denoise limit')).toHaveAttribute('aria-posinset', '2');
    const dot = treeitem('Noise').querySelector<HTMLButtonElement>('button[aria-label="Toggle active"]')!;
    await act(() => userEvent.click(dot));
    expect(output('mask-modifiers')).toBe('noise:off:0.25,denoise:on:0.8');
    treeitem('Denoise limit').focus();
    await act(() => userEvent.keyboard('{Delete}'));
    expect(output('mask-modifiers')).toBe('noise:off:0.25');
    expect(document.activeElement).toBe(treeitem('Mask'));
  });

  it('projects adjustment rows in stack order; the dot toggles one entry', async () => {
    await renderTree([
      layerContract('r1', 'raster', {
        adjustments: [
          { brightness: 0.2, contrast: 0, id: 'a1', isEnabled: true, type: 'brightness-contrast' },
          { id: 'a2', isEnabled: true, saturation: -0.4, type: 'hsl' },
          { curves: {}, id: 'a3', isEnabled: true, type: 'curves' },
        ],
        name: 'Painting',
      }),
    ]);
    expect(treeitem('Brightness/Contrast')).toHaveAttribute('aria-posinset', '1');
    expect(treeitem('Saturation')).toHaveAttribute('aria-posinset', '2');
    expect(treeitem('Curves')).toHaveAttribute('aria-setsize', '3');
    expect(treeitem('Saturation').textContent).toContain('-40%');
    const dot = treeitem('Saturation').querySelector<HTMLButtonElement>('button[aria-label="Toggle active"]')!;
    await act(() => userEvent.click(dot));
    expect(output('raster-adjustments')).toBe('a1:on,a2:off,a3:on');
  });

  it('renames an adjustment entry inline with F2 and restores the kind name on an empty draft', async () => {
    await renderTree([
      layerContract('r1', 'raster', {
        adjustments: [{ id: 'a1', isEnabled: true, saturation: 0.1, type: 'hsl' }],
        name: 'Painting',
      }),
    ]);
    treeitem('Saturation').focus();
    await act(() => userEvent.keyboard('{F2}'));
    const input = host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    await act(() => userEvent.clear(input));
    await act(() => userEvent.type(input, 'Pop'));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(treeitem('Pop')).toBeTruthy();
    expect(treeitem('Pop').textContent).toContain('+10%');

    treeitem('Pop').focus();
    await act(() => userEvent.keyboard('{F2}'));
    const again = host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    await act(() => userEvent.clear(again));
    await act(() => userEvent.keyboard('{Enter}'));
    expect(treeitem('Saturation')).toBeTruthy();
  });

  it('contains live preview ticks to the edited layer and its child rows', async () => {
    await renderTree([
      layerContract('r1', 'raster', {
        adjustments: [{ id: 'a1', isEnabled: true, saturation: 0, type: 'hsl' }],
        name: 'Painting',
      }),
      layerContract('r2', 'raster', { name: 'Bystander' }),
      layerContract('rg', 'regional_guidance', { name: 'Region' }),
    ]);
    resetLayerRowCommits();
    for (let tick = 1; tick <= 30; tick++) {
      await act(() =>
        dispatchExternal({
          config: {
            adjustments: [{ id: 'a1', isEnabled: true, saturation: tick / 100, type: 'hsl' }],
            layerType: 'raster',
          },
          id: 'r1',
          type: 'updateCanvasLayerConfig',
        })
      );
    }
    const commits = getLayerRowCommits();
    expect(Object.keys(commits).filter((id) => id !== 'r1' && !id.startsWith('child:r1:'))).toEqual([]);
    expect(commits['child:r1:a1']).toBe(30);
    expect(treeitem('Saturation').textContent).toContain('+30%');
  });

  it('reorders adjustment entries with a pointer drag', async () => {
    await renderTree([
      layerContract('r1', 'raster', {
        adjustments: [
          { brightness: 0.2, contrast: 0, id: 'a1', isEnabled: true, type: 'brightness-contrast' },
          { curves: {}, id: 'a2', isEnabled: true, type: 'curves' },
        ],
        name: 'Painting',
      }),
    ]);
    const curves = host!.querySelector<HTMLElement>('[data-layer-row-id="child:r1:a2"]')!;
    const bc = host!.querySelector<HTMLElement>('[data-layer-row-id="child:r1:a1"]')!;
    const start = centre(curves);
    const end = centre(bc);
    await act(() => pointer('pointerdown', curves, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, end.x, end.y - 10));
    await act(() => pointer('pointerup', document, end.x, end.y - 10));
    expect(output('raster-adjustments')).toBe('a2:on,a1:on');
  });

  it('moves a reference image to another regional layer with a pointer drag', async () => {
    await renderTree([
      layerContract('rg', 'regional_guidance', { name: 'Region A', referenceImages: [referenceImage('ref1')] }),
      layerContract('rg2', 'regional_guidance', { name: 'Region B' }),
    ]);
    const refRow = host!.querySelector<HTMLElement>('[data-layer-row-id="child:rg:ref1"]')!;
    const targetRow = host!.querySelector<HTMLElement>('[data-layer-row-id="rg2"]')!;
    const start = centre(refRow);
    const end = centre(targetRow);
    await act(() => pointer('pointerdown', refRow, start.x, start.y));
    await act(() => pointer('pointermove', document, start.x + 8, start.y));
    await act(() => pointer('pointermove', document, end.x, end.y));
    await act(() => pointer('pointerup', document, end.x, end.y));
    expect(output('regional-refs')).toBe('rg[] rg2[ref1:on]');
    expect(output('selected-layer')).toBe('rg2');
    expect(getLayerChildSelection()).toMatchObject({ itemId: 'ref1', layerId: 'rg2' });
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
    /** Drags `from` to `to`, landing `dy` pixels off its centre. */
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

    it('appends to a raster layer or group row it is dropped on', async () => {
      await renderTree(adjustedLayers());
      await drag(row('child:r1:a1'), row('g1'));
      expect(output('adjustment-owners')).toBe('r1[a2] r2[b1] g1[a1]');
      await drag(row('child:r1:a2'), row('r2'));
      expect(output('adjustment-owners')).toBe('r2[b1,a2] g1[a1]');
    });

    it('reveals and focuses an entry moved from the menu into a collapsed group member', async () => {
      await renderTree(adjustedLayers());
      expect(host!.querySelector('[data-layer-row-id="r3"]')).toBeNull();
      const child = { itemId: 'a2', key: 'child:r1:a2', kind: 'adjustment-hsl', layerId: 'r1', stack: 'raster' };
      await act(() => surfaceCommands.current!.moveChildToLayer(child as ProjectedChildRow, 'r3'));
      await settle();
      expect(output('adjustment-owners')).toBe('r1[a1] r2[b1] r3[a2]');
      expect(output('selected-layer')).toBe('r3');
      expect(getLayerChildSelection()).toMatchObject({ itemId: 'a2', layerId: 'r3' });
      expect(readLayerPanelState(PROJECT_ID, 'r3')).toMatchObject({ focusId: 'child:r3:a2', selectedIds: ['r3'] });
      expect(document.activeElement).toBe(row('child:r3:a2'));
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

  it('walks child rows from the keyboard and routes their context menu', async () => {
    await renderTree(regionWithRefs());
    treeitem('Region').focus();
    await act(() => userEvent.keyboard('{ArrowRight}'));
    expect(document.activeElement).toBe(treeitem('Reference image 1'));
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(document.activeElement).toBe(treeitem('Region'));
    await act(() => userEvent.keyboard('{ArrowLeft}'));
    expect(treeitem('Region')).toHaveAttribute('aria-expanded', 'false');
    await act(() => userEvent.keyboard('{ArrowRight}{ArrowRight}'));
    await act(() => userEvent.keyboard('{Shift>}{F10}{/Shift}'));
    expect(output('surface')).toBe('child-menu:child:rg:ref1');
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

  it('keeps the tree item focused when one of its controls is clicked, and lets a rename blur follow a click', async () => {
    await renderTree(manyLayers(3));
    await act(() => userEvent.click(treeitem('Layer 1').querySelector('button[aria-label="Toggle lock"]')!));
    expect(document.activeElement).toBe(treeitem('Layer 1'));
    expect(tabStops()).toEqual([treeitem('Layer 1')]);
    await act(() => userEvent.keyboard('{F2}'));
    await act(() => userEvent.keyboard('New'));
    await act(() => userEvent.click(treeitem('Layer 2')));
    expect(treeitem('New')).not.toBeNull();
    expect(document.activeElement).toBe(treeitem('Layer 2'));
    expect(output('selected-layer')).toBe('l2');
  });

  it('carries the selection with a keyboard move', async () => {
    await renderTree(manyLayers(3));
    await act(() => userEvent.click(treeitem('Layer 0')));
    await act(() => userEvent.click(treeitem('Layer 1'), { modifiers: ['Shift'] }));
    expect(output('selected-layers')).toBe('l0,l1');
    treeitem('Layer 0').focus();
    await act(() => userEvent.keyboard('{Alt>}{ArrowDown}{/Alt}'));
    expect(output('layer-order')).toBe('l2,l0,l1');
  });

  it('leaves a locked group closed when an indent into it is refused, and ignores an empty move', async () => {
    await renderTree([
      groupContract('g', [paint('inner', 'Inner')], { isLocked: true, name: 'Group' }),
      paint('bottom', 'Bottom'),
    ]);
    treeitem('Bottom').focus();
    await act(() => userEvent.keyboard('{Alt>}{ArrowRight}{/Alt}'));
    expect(output('layer-order')).toBe('inner,bottom');
    expect(readLayerPanelState(PROJECT_ID, null).expandedGroupIds).toEqual([]);
    await act(() => userEvent.keyboard('{ArrowUp}{ArrowRight}{ArrowDown}'));
    expect(document.activeElement).toBe(treeitem('Inner'));
    await act(() => userEvent.click(treeitem('Group')));
    await act(() => userEvent.click(treeitem('Inner'), { modifiers: ['Control'] }));
    treeitem('Inner').focus();
    await act(() => userEvent.keyboard('{Alt>}{ArrowLeft}{/Alt}'));
    expect(output('layer-order')).toBe('inner,bottom');
  });

  it('renames from the keyboard, reseeds the draft each time, and hands focus back to the row', async () => {
    await renderTree(manyLayers(3));
    treeitem('Layer 1').focus();
    await act(() => userEvent.keyboard('{F2}'));
    const input = host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    expect(input.value).toBe('Layer 1');
    await act(() => userEvent.keyboard('abc{Escape}'));
    expect(document.activeElement).toBe(treeitem('Layer 1'));
    await act(() => userEvent.keyboard('{F2}'));
    expect(host!.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!.value).toBe('Layer 1');
    await act(() => userEvent.keyboard('Renamed{Enter}'));
    expect(treeitem('Renamed')).not.toBeNull();
    expect(document.activeElement).toBe(treeitem('Renamed'));
  });
});

describe('LayersTree color labels', () => {
  it('shows a color-label strip on labelled rows only', async () => {
    await renderTree([
      layerContract('tagged', 'raster', { colorLabel: 'red', name: 'Tagged' }),
      groupContract('folder', [paint('inside', 'Inside')], { colorLabel: 'blue' }),
      paint('plain', 'Plain'),
    ]);
    expect(host!.querySelector('[data-layer-row-id="tagged"] [data-color-label="red"]')).not.toBeNull();
    expect(host!.querySelector('[data-layer-row-id="folder"] [data-color-label="blue"]')).not.toBeNull();
    expect(host!.querySelector('[data-layer-row-id="plain"] [data-color-label]')).toBeNull();
  });
});
