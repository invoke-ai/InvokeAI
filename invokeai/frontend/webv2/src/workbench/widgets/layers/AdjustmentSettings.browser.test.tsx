import type {
  CanvasAdjustmentEntry,
  CanvasGroupContract,
  CanvasNodeContract,
  PreparedDocumentEdit,
} from '@workbench/canvas-engine/api';
import type { StructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
/* oxlint-disable react-perf/jsx-no-new-function-as-prop */

import { ChakraProvider } from '@chakra-ui/react';
import { applyThemeToRoot } from '@theme/applyTheme';
import { system } from '@theme/system';
import { createStructuralEngineStub } from '@workbench/canvas-engine/controllers/structuralEngine.testStub';
import { groupContract, layerContract } from '@workbench/canvas-engine/document-model/documentFixtures.testStub';
import { getDocumentIndex } from '@workbench/canvas-engine/document/documentIndex';
import { commitPreparedEdit } from '@workbench/widgets/canvas/useStructuralCommit';
import { createInstance } from 'i18next';
import { act, type ReactElement, useSyncExternalStore } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { I18nextProvider, initReactI18next } from 'react-i18next';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { userEvent } from 'vitest/browser';

import { type AdjustmentOwner, type AdjustmentsEngine, AdjustmentSettings } from './AdjustmentSettings';
import { createFakeAdjustmentStatus } from './AdjustmentStatusIcon.testing';
import { CURVE_SIZE } from './curveEditorMath';
import { LayerGroupContextMenu, type LayerGroupContextMenuEngine } from './LayerGroupContextMenu';

const i18n = createInstance();
void i18n.use(initReactI18next).init({ fallbackLng: 'en', initAsync: false, lng: 'en', resources: {} });

const noopDispatch = (): void => undefined;
vi.mock('@workbench/useCanvasProjectMutationDispatch', () => ({
  useCanvasProjectMutationDispatch: () => noopDispatch,
}));
const notify = vi.hoisted(() => ({ error: vi.fn(), info: vi.fn(), success: vi.fn() }));
vi.mock('@workbench/useNotify', () => ({ useNotify: () => notify }));
vi.mock('@workbench/WorkbenchContext', () => ({
  useActiveProjectId: () => 'project-1',
  useActiveProjectSelector: <T,>(select: (project: { canvas: { document: unknown } }) => T): T =>
    select({ canvas: { document: stub.document() } }),
}));

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement | null = null;
let root: Root | null = null;

const initialEntries = (): CanvasAdjustmentEntry[] => [
  { brightness: 0.1, contrast: 0, id: 'bc1', isEnabled: true, type: 'brightness-contrast' },
  { curves: {}, id: 'cv1', isEnabled: true, type: 'curves' },
  { id: 'hue1', isEnabled: true, rotation: 0, type: 'hue' },
  { gamma: 1, id: 'lv1', inBlack: 0, inWhite: 255, isEnabled: true, outBlack: 0, outWhite: 255, type: 'levels' },
  {
    amount: 0.1,
    distribution: 'gaussian',
    grain: 'monochrome',
    id: 'nz1',
    isEnabled: true,
    response: 'midtones',
    seed: 42,
    size: 1,
    type: 'noise',
  },
];

type Owner = 'raster' | 'group';
const OWNER_ID: Record<Owner, string> = { group: 'group-1', raster: 'layer-1' };

const ownerNode = (owner: Owner, adjustments: CanvasAdjustmentEntry[]): CanvasNodeContract =>
  owner === 'group'
    ? groupContract('group-1', [], { adjustments, name: 'Group 1' })
    : layerContract('layer-1', 'raster', { adjustments, name: 'Layer 1' });

let stub: StructuralEngineStub;
let adjustmentStatus = createFakeAdjustmentStatus();
let ownerId = OWNER_ID.raster;
const documentOwner = (): AdjustmentOwner =>
  getDocumentIndex(stub.document()).byId.get(ownerId)!.node as unknown as AdjustmentOwner;
const latestAdjustments = (): readonly CanvasAdjustmentEntry[] => documentOwner().adjustments ?? [];
const commits = () => stub.commits;

const landConcurrentEdit = (update: (adjustments: readonly CanvasAdjustmentEntry[]) => CanvasAdjustmentEntry[]) =>
  settle(() => {
    const outcome = commitPreparedEdit(stub.engine, 'Edit', (model) =>
      model.prepare({
        config: { adjustments: update(latestAdjustments()), layerType: documentOwner().type },
        id: ownerId,
        type: 'patch-config',
      })
    );
    expect(outcome).toEqual({ status: 'committed' });
  });

const Harness = ({ entryId, withEngine }: { entryId: string; withEngine: boolean }) => {
  const layer = useSyncExternalStore(stub.subscribe, documentOwner);
  return (
    <AdjustmentSettings
      engine={withEngine ? (stub.engine as unknown as AdjustmentsEngine) : null}
      entryId={entryId}
      layer={layer}
    />
  );
};

const settle = (action: () => void): Promise<void> =>
  act(async () => {
    action();
    await new Promise<void>((resolve) => {
      globalThis.setTimeout(resolve, 30);
    });
  });

const setUp = (owner: Owner, adjustments: CanvasAdjustmentEntry[]): void => {
  ownerId = OWNER_ID[owner];
  stub = createStructuralEngineStub({
    layers: [ownerNode(owner, adjustments)],
    schedulePreview: (flush) => {
      flush();
      return () => undefined;
    },
  });
  adjustmentStatus = createFakeAdjustmentStatus();
  Object.assign(stub.engine, { interaction: adjustmentStatus.interaction });
};

const mount = async (element: ReactElement): Promise<void> => {
  applyThemeToRoot('classic');
  host = document.createElement('div');
  host.style.width = '260px';
  document.body.append(host);
  root = createRoot(host);
  await settle(() => {
    root?.render(
      <I18nextProvider i18n={i18n}>
        <ChakraProvider value={system}>{element}</ChakraProvider>
      </I18nextProvider>
    );
  });
};

const render = async (entryId = 'cv1', owner: Owner = 'raster', adjustments = initialEntries(), withEngine = true) => {
  setUp(owner, adjustments);
  await mount(<Harness entryId={entryId} withEngine={withEngine} />);
  return host!.querySelector<SVGSVGElement>(`svg[viewBox="0 0 ${CURVE_SIZE} ${CURVE_SIZE}"]`)!;
};

const handles = (svg: SVGSVGElement): SVGCircleElement[] => Array.from(svg.querySelectorAll('circle'));

const centreOf = (element: Element): { x: number; y: number } => {
  const rect = element.getBoundingClientRect();
  return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
};

const pointer = (target: EventTarget, type: string, x: number, y: number): void => {
  target.dispatchEvent(
    new PointerEvent(type, { bubbles: true, button: 0, clientX: x, clientY: y, isPrimary: true, pointerId: 1 })
  );
};

afterEach(async () => {
  await settle(() => root?.unmount());
  host?.remove();
  host = null;
  root = null;
  vi.clearAllMocks();
});

const scrubber = (label: string): HTMLElement => {
  const frame = [...host!.querySelectorAll<HTMLElement>('[data-scope="scrubber"]')].find(
    (candidate) => candidate.querySelector('[data-part="label"]')?.textContent === label
  );
  if (!frame) {
    throw new Error(`No scrubber labelled ${label}`);
  }
  return frame;
};

const scrub = async (frame: HTMLElement, offsets: number[]): Promise<void> => {
  const rect = frame.getBoundingClientRect();
  const startX = rect.left + rect.width / 2;
  const y = rect.top + rect.height / 2;
  // The track is inset 6px from each edge of the frame.
  const xAt = (offset: number): number => startX + offset * (rect.width - 12);
  await settle(() => pointer(frame, 'pointerdown', startX, y));
  for (const offset of offsets) {
    await settle(() => pointer(window, 'pointermove', xAt(offset), y));
  }
  await settle(() => pointer(window, 'pointerup', xAt(offsets.at(-1) ?? 0), y));
};

const inverseEntries = (edit: PreparedDocumentEdit): unknown =>
  (edit.inverse as unknown as { config: { adjustments?: unknown } }).config.adjustments;

const forwardEntries = (edit: PreparedDocumentEdit): CanvasAdjustmentEntry[] =>
  (edit.forward as unknown as { config: { adjustments: CanvasAdjustmentEntry[] } }).config.adjustments;

describe('group-owned adjustment editors', () => {
  it('previews and commits through the group config arm', async () => {
    await render('hue1', 'group');
    await scrub(scrubber('widgets.layers.adjustments.hue'), [0.25]);

    expect(commits()).toHaveLength(1);
    const forward = commits()[0]!.forward as unknown as { config: { layerType: string }; id: string };
    expect(forward.config.layerType).toBe('group');
    expect(forward.id).toBe('group-1');
    expect((forwardEntries(commits()[0]!)[2] as { rotation: number }).rotation).toBe(90);
  });
});

describe('scalar and levels editors', () => {
  it('previews every step of a hue drag and records the drag as one whole-stack patch', async () => {
    await render('hue1');
    const hue = scrubber('widgets.layers.adjustments.hue');
    const rect = hue.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    const trackWidth = rect.width - 12;

    await settle(() => pointer(hue, 'pointerdown', startX, y));
    await settle(() => pointer(window, 'pointermove', startX + trackWidth * 0.1, y));
    await settle(() => pointer(window, 'pointermove', startX + trackWidth * 0.25, y));

    expect(commits()).toHaveLength(0);
    expect(latestAdjustments()[2]).toMatchObject({ id: 'hue1', rotation: 90 });

    await settle(() => pointer(window, 'pointerup', startX + trackWidth * 0.25, y));

    expect(commits()).toHaveLength(1);
    const forward = forwardEntries(commits()[0]!);
    expect(forward[2]).toMatchObject({ id: 'hue1', rotation: 90, type: 'hue' });
    expect(forward[0]).toEqual(initialEntries()[0]);
    expect(inverseEntries(commits()[0]!)).toEqual(initialEntries());
    await stub.engine.history.undo();
    expect(latestAdjustments()).toEqual(initialEntries());
  });

  it('keeps an edit that lands mid-drag and undoes only the dragged field', async () => {
    await render('hue1');
    const hue = scrubber('widgets.layers.adjustments.hue');
    const rect = hue.getBoundingClientRect();
    const startX = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    const trackWidth = rect.width - 12;

    await settle(() => pointer(hue, 'pointerdown', startX, y));
    await settle(() => pointer(window, 'pointermove', startX + trackWidth * 0.1, y));
    await landConcurrentEdit((adjustments) =>
      adjustments.map((candidate) =>
        candidate.id === 'hue1'
          ? { ...candidate, isEnabled: false }
          : candidate.id === 'bc1'
            ? { ...candidate, contrast: 0.5 }
            : candidate
      )
    );

    expect(latestAdjustments()[2]).toMatchObject({ isEnabled: false, rotation: 0 });

    await settle(() => pointer(window, 'pointermove', startX + trackWidth * 0.25, y));

    expect(latestAdjustments()[2]).toMatchObject({ isEnabled: false, rotation: 90 });

    await settle(() => pointer(window, 'pointerup', startX + trackWidth * 0.25, y));

    const forward = forwardEntries(commits().at(-1)!);
    expect(forward[2]).toMatchObject({ id: 'hue1', isEnabled: false, rotation: 90 });
    expect(forward[0]).toMatchObject({ contrast: 0.5, id: 'bc1' });
    const inverse = inverseEntries(commits().at(-1)!) as CanvasAdjustmentEntry[];
    expect(inverse[2]).toMatchObject({ id: 'hue1', isEnabled: false, rotation: 0 });
    expect(inverse[0]).toMatchObject({ contrast: 0.5, id: 'bc1' });
    await stub.engine.history.undo();
    expect(latestAdjustments()[2]).toMatchObject({ isEnabled: false, rotation: 0 });
    expect(latestAdjustments()[0]).toMatchObject({ contrast: 0.5 });
  });

  it('records nothing and restores the stack when a scrub ends where it started', async () => {
    await render('hue1');
    await scrub(scrubber('widgets.layers.adjustments.hue'), [0.25, 0]);

    expect(commits()).toHaveLength(0);
    expect(latestAdjustments()).toEqual(initialEntries());
  });
});
const NOISE_INDEX = 4;

const segment = (label: string): HTMLElement => {
  const item = [...host!.querySelectorAll<HTMLElement>('[data-scope="segment-group"][data-part="item"]')].find(
    (candidate) => candidate.textContent?.trim() === label
  );
  if (!item) {
    throw new Error(`No segment labelled ${label}`);
  }
  return item;
};

describe('noise editor', () => {
  it("shows the entry's choices and commits a distribution change as one undoable patch", async () => {
    await render('nz1');
    expect(host!.querySelector('[aria-label="widgets.layers.adjustments.noiseDistribution"]')).not.toBeNull();
    expect(segment('widgets.layers.adjustments.noiseDistributions.gaussian').dataset.state).toBe('checked');
    expect(segment('widgets.layers.adjustments.noiseGrains.monochrome').dataset.state).toBe('checked');
    expect(segment('widgets.layers.adjustments.noiseResponses.midtones').dataset.state).toBe('checked');
    expect(host!.querySelector('[aria-label="widgets.layers.adjustments.noiseResponse"]')).not.toBeNull();

    await settle(() => segment('widgets.layers.adjustments.noiseDistributions.uniform').click());

    expect(commits()).toHaveLength(1);
    expect(forwardEntries(commits()[0]!)[NOISE_INDEX]).toEqual({
      ...initialEntries()[NOISE_INDEX],
      distribution: 'uniform',
    });
    expect(inverseEntries(commits()[0]!)).toEqual(initialEntries());
    await stub.engine.history.undo();
    expect(latestAdjustments()).toEqual(initialEntries());
  });
});

describe('curves entry editor', () => {
  it('moves a handle while dragging and commits the whole stack with the pre-gesture inverse', async () => {
    const svg = await render();
    const target = handles(svg).at(-1)!;
    const before = Number(target.getAttribute('cy'));
    const start = centreOf(target);

    await settle(() => pointer(target, 'pointerdown', start.x, start.y));
    await settle(() => pointer(target, 'pointermove', start.x, start.y + 40));

    const during = Number(handles(svg).at(-1)!.getAttribute('cy'));
    expect(during).toBeGreaterThan(before);

    await settle(() => pointer(target, 'pointerup', start.x, start.y + 40));
    expect(Number(handles(svg).at(-1)!.getAttribute('cy'))).toBeCloseTo(during, 5);
    expect(commits()).toHaveLength(1);
    const edit = commits()[0]!;
    expect(edit.inverse).toMatchObject({ id: 'layer-1', type: 'updateCanvasLayerConfig' });
    expect((edit.inverse as { config: { adjustments?: unknown } }).config.adjustments).toEqual(initialEntries());
    const forward = (edit.forward as unknown as { config: { adjustments: CanvasAdjustmentEntry[] } }).config
      .adjustments;
    expect(forward[0]).toEqual(initialEntries()[0]);
    expect(forward[1]).toMatchObject({ id: 'cv1', type: 'curves' });
    expect((forward[1] as { curves: { r?: unknown[] } }).curves.r).toBeDefined();
  });

  it('restores the pre-drag stack and records nothing when the drag is cancelled', async () => {
    const svg = await render();
    const target = handles(svg).at(-1)!;
    const before = Number(target.getAttribute('cy'));
    const start = centreOf(target);

    await settle(() => pointer(target, 'pointerdown', start.x, start.y));
    await settle(() => pointer(target, 'pointermove', start.x, start.y + 40));
    await settle(() => pointer(target, 'pointercancel', start.x, start.y + 40));

    expect(Number(handles(svg).at(-1)!.getAttribute('cy'))).toBeCloseTo(before, 5);
    expect(commits()).toHaveLength(0);
    expect(latestAdjustments()).toEqual(initialEntries());
  });
});

const MENU_ANCHOR = { height: 1, width: 1, x: 300, y: 10 };
const closeMenu = (): void => undefined;

describe('spatial adjustment editors', () => {
  type SpatialEntry = Extract<CanvasAdjustmentEntry, { type: 'clarity' | 'texture' | 'dehaze' }>;

  const MenuHarness = () => {
    const owner = useSyncExternalStore(stub.subscribe, documentOwner);
    const last = owner.adjustments?.at(-1);
    return (
      <>
        <LayerGroupContextMenu
          anchor={MENU_ANCHOR}
          editingLocked={false}
          engine={stub.engine as unknown as LayerGroupContextMenuEngine}
          group={owner as unknown as CanvasGroupContract}
          stack="raster"
          onClose={closeMenu}
        />
        {last ? (
          <AdjustmentSettings engine={stub.engine as unknown as AdjustmentsEngine} entryId={last.id} layer={owner} />
        ) : null}
      </>
    );
  };

  const sliderOf = (label: string): HTMLElement => scrubber(label).querySelector<HTMLElement>('[role="slider"]')!;

  it('adds Clarity from the menu at identity, then scrubs and resets its amount as undoable steps', async () => {
    setUp('group', []);
    await mount(<MenuHarness />);

    const trigger = document.querySelector<HTMLElement>('[aria-label="widgets.layers.menu.addAdjustment"]')!;
    await act(() => userEvent.click(trigger));
    const clarityItem = () => document.querySelector<HTMLElement>('[role="menuitem"][data-value="add-clarity"]');
    await expect.poll(clarityItem).not.toBeNull();
    const clarity = clarityItem()!;
    expect(clarity.textContent).toBe('widgets.layers.adjustments.clarity');
    await act(() => userEvent.click(clarity));
    await settle(() => undefined);

    expect(commits()).toHaveLength(1);
    const [added] = latestAdjustments() as SpatialEntry[];
    expect(added).toEqual({ amount: 0, id: expect.any(String), isEnabled: true, size: 40, type: 'clarity' });
    expect(sliderOf('widgets.layers.adjustments.amount').getAttribute('aria-valuetext')).toBe('0');
    expect(sliderOf('widgets.layers.adjustments.size').getAttribute('aria-valuetext')).toBe('40 px');

    await scrub(scrubber('widgets.layers.adjustments.amount'), [0.1, 0.25]);
    expect(commits()).toHaveLength(2);
    expect(latestAdjustments()).toEqual([{ ...added, amount: 100 }]);
    expect(sliderOf('widgets.layers.adjustments.amount').getAttribute('aria-valuetext')).toBe('+100');

    await act(async () => {
      sliderOf('widgets.layers.adjustments.amount').focus();
      await userEvent.keyboard('{Backspace}');
    });
    expect(commits()).toHaveLength(3);
    expect(latestAdjustments()).toEqual([added]);

    await stub.engine.history.undo();
    expect(latestAdjustments()).toEqual([{ ...added, amount: 100 }]);
  });
});
