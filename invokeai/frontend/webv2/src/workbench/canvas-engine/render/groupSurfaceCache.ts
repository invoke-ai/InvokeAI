/**
 * Document-space group composites keyed by scope shape, member appearance, effective matrices and in-scope
 * previews. Own opacity/blend apply at draw time. A stable slot refreshes in place: only the region its members
 * damaged since the last draw is cleared and redrawn, and geometry changes redraw without reallocating when the
 * size holds. Document-resolution buffers resample transformed members twice and upscale above 100% zoom. A scope
 * whose own stack is spatial keeps its raw composite and shows a separate adjusted surface, evaluated
 * asynchronously by the shared spatial display rules (`spatialDisplay.ts`).
 */

import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import {
  adjustmentsKey,
  applyAdjustments,
  hasSpatialAdjustments,
  isIdentityAdjustments,
} from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';
import {
  createSpatialDisplay,
  type AdjustedSurfaceStatus,
  type DisplayedPixelsState,
  type SpatialDisplay,
  type SpatialDisplayOptions,
  type SpatialDisplayTarget,
  worseDisplayedState,
} from '@workbench/canvas-engine/render/spatialDisplay';

import type { GroupCompositeScope, GroupSurfaceContent } from './groupCompositeScopes';

export interface GroupSurfaceResult {
  readonly surface: RasterSurface;
  /** Document-space placement of the surface. */
  readonly rect: Rect;
  /** How final the pixels are: the group's own spatial stack and every member's and nested group's included. */
  readonly state: DisplayedPixelsState;
}

/** A member's displayed pixels: its own adjusted surface (null for raw) and how final they are. */
export interface MemberDisplayedPixels {
  readonly surface: RasterSurface | null;
  readonly state: DisplayedPixelsState;
}

export interface GroupSurfaceDeps {
  createSurface(width: number, height: number): RasterSurface;
  /** Receives the running byte total after every allocation, resize and release. */
  onBytesChange?(bytes: number): void;
  getCacheEntry(layerId: string): LayerCacheEntry | undefined;
  /** Surface-local damage a member's cache received since `version`, or null when unknown. */
  damageSince(layerId: string, version: number): Rect | null;
  /** The member's own displayed pixels (its personal stack applied). */
  getAdjustedSurface(layer: CanvasRasterLayerContractV2, entry: LayerCacheEntry): MemberDisplayedPixels;
  diagnostics?: CanvasDiagnostics;
  /**
   * Evaluates group stacks with spatial entries, reporting by group id. Without it those scopes show the per-pixel
   * part only.
   */
  spatial?: SpatialDisplayOptions;
}

export interface GroupSurfaceCache {
  /** Total RGBA bytes the cached group surfaces hold, for the memory budget. */
  byteSize(): number;
  /**
   * `frame` marks the canvas frame's reads: a group reports, under its id in `adjustmentStatus`, the status of the
   * slot the frame read last (else of its most recently read slot), so reads by other consumers with other keys
   * (the overview, picking) never flip it.
   */
  get(
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    options?: { readonly frame?: boolean }
  ): GroupSurfaceResult | null;
  /** Drops every cached group not named; call when document structure changes. */
  prune(liveGroupIds: ReadonlySet<string>): void;
  /**
   * Makes the named groups redraw in full on their next read, keeping their surfaces: a member's displayed pixels
   * changed without a cache version (a spatial adjustment landed). A spatial group's adjusted surface keeps its
   * pixels outside member content damage until its own job lands.
   */
  invalidate(groupIds: Iterable<string>): void;
  /** Drops the spatial slots of every group `keep` rejects, e.g. a group whose stack lost its spatial entries. */
  retainSpatial(keep: (groupId: string) => boolean): void;
  /**
   * Abandons running and scheduled group jobs; those slots show a whole per-pixel interim on their next read and start
   * again. Exact and failed slots keep their state.
   */
  cancelSpatial(): void;
  /** The access clock; a slot read at or after a captured tick is in use by that frame. */
  tick(): number;
  /** Evicts slots least-recently-used first until within budget, keeping slots read after `sinceTick`. */
  evict(budgetBytes: number, sinceTick: number): number;
  clear(): void;
}

const matKey = (m: Mat2d): string => `${m.a},${m.b},${m.c},${m.d},${m.e},${m.f}`;
const rectKey = (r: Rect): string => `${r.x},${r.y},${r.width},${r.height}`;
const sameRect = (a: Rect, b: Rect): boolean =>
  a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;

// A changed source pixel reaches one source pixel further through bilinear sampling, whatever the member's scale,
// plus one destination pixel of edge antialiasing.
const SOURCE_FILTER_RADIUS = 1;
const ANTIALIAS_MARGIN = 1;

// The frame composites with live session matrices while the overview composites
// the settled contract, so one group legitimately holds two keys at once; two
// slots stop those consumers evicting each other every tick.
const SLOTS_PER_GROUP = 2;

/** Raw repaints a spatial slot records apart; older ones fold into the oldest entry, coarser but still known. */
const RAW_DAMAGE_TRAIL_LIMIT = 16;

const NO_AREA: Rect = { height: 0, width: 0, x: 0, y: 0 };

/** A spatial scope's raw composite history and the adjusted surface the display shows. */
interface SpatialGroupState {
  /** Bumped by every raw repaint. */
  rawVersion: number;
  /** Surface-local damage of the raw repaints from `first` to `version`, `null` when unknown; oldest first. */
  readonly trail: { readonly first: number; readonly version: number; readonly damage: Rect | null }[];
  readonly adjusted: RasterSurface;
  /** Raw version and stack key `adjusted` was last brought to; null forces a whole per-pixel interim. */
  adjustedFor: { readonly version: number; readonly key: string } | null;
}

/**
 * How a spatial scope's member or nested group draws into the raw composite. A change repaints only its bounds,
 * instead of moving the scope to another slot whose adjusted copy would restart from a whole per-pixel interim.
 */
interface DrawnAppearance {
  /** Opacity and blend, plus the matrix for a member. */
  readonly placement: string;
  /** `adjustmentsKey` of its own stack. */
  readonly stack: string;
  /** A spatial stack's display keeps its pixels through a parameter edit until its result lands (`invalidate`). */
  readonly spatial: boolean;
  /** Document-space bounds it draws into. */
  readonly bounds: Rect | null;
}

interface GroupSlot {
  readonly groupId: string;
  /** This slot's spatial display owner id; each slot runs its own jobs. */
  readonly owner: string;
  shapeKey: string;
  rect: Rect;
  readonly surface: RasterSurface;
  /** Cache version of every member pixel source the surface currently shows. */
  versions: ReadonlyMap<string, number>;
  /** Placement of the floating selection the surface shows, so a drag refreshes in place. */
  float: DrawnFloat | null;
  lastUsed: number;
  /** Member pixels changed in a way damage tracking cannot see; the next read redraws everything. */
  stale: boolean;
  /** How final the members and nested groups drawn at the last repaint were. */
  inputs: DisplayedPixelsState;
  /** Present while the slot holds a spatial scope's raw composite. */
  spatial: SpatialGroupState | null;
  /** A spatial scope's shape without its members' looks, and those looks; null for other scopes. */
  structureKey: string | null;
  appearances: ReadonlyMap<string, DrawnAppearance> | null;
}

interface DrawnFloat {
  readonly placement: string;
  /** Document-space bounds the float's resampled pixels can reach. */
  readonly landing: Rect;
}

/** What changed in a slot's members since its last draw (document space). */
interface Damage {
  /** Pixels that changed. */
  readonly region: Rect | null;
  /** Looks to redraw whose pixels did not change. */
  readonly reread: Rect | null;
}

interface DrawnEntry {
  readonly leaf: SemanticLeaf;
  readonly matrix: Mat2d;
  readonly entry: LayerCacheEntry | null;
  readonly bounds: Rect | null;
  readonly float: DrawnFloat | null;
}

const unionOf = (a: Rect | null, b: Rect | null): Rect | null => (a && b ? union(a, b) : (a ?? b));

const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * 4;
const slotBytes = (slot: GroupSlot): number =>
  surfaceBytes(slot.surface) + (slot.spatial ? surfaceBytes(slot.spatial.adjusted) : 0);

/** Union of the raw damage since `version`, or null when unknown or older than the trail. */
const rawDamageSince = (spatial: SpatialGroupState, version: number): Rect | null => {
  const { trail } = spatial;
  if (trail.length === 0 || trail[0]!.first > version + 1) {
    return null;
  }
  let damaged = NO_AREA;
  for (const step of trail) {
    if (step.version <= version) {
      continue;
    }
    if (!step.damage) {
      return null;
    }
    damaged = union(damaged, step.damage);
  }
  return damaged;
};

export const createGroupSurfaceCache = (deps: GroupSurfaceDeps): GroupSurfaceCache => {
  const cache = new Map<string, GroupSlot[]>();
  const objectIds = new WeakMap<object, number>();
  const slotsByOwner = new Map<string, GroupSlot>();
  /** The slot each group's latest frame read used; its status is the one the group reports. */
  const frameSlots = new Map<string, GroupSlot>();
  /** Last status reported per group. */
  const reported = new Map<string, AdjustedSurfaceStatus>();
  let nextObjectId = 1;
  let nextSlot = 1;
  let clock = 0;
  let totalBytes = 0;

  const objectId = (value: object): number => {
    let id = objectIds.get(value);
    if (id === undefined) {
      id = nextObjectId++;
      objectIds.set(value, id);
    }
    return id;
  };

  const adjustBytes = (delta: number): void => {
    if (delta === 0) {
      return;
    }
    totalBytes += delta;
    deps.onBytesChange?.(totalBytes);
  };

  const reportStatus = (groupId: string): void => {
    const reporting = frameSlots.get(groupId) ?? cache.get(groupId)?.[0];
    const status = reporting?.spatial ? (display?.status(reporting.owner) ?? null) : null;
    if ((reported.get(groupId) ?? null) === status) {
      return;
    }
    if (status) {
      reported.set(groupId, status);
    } else {
      reported.delete(groupId);
    }
    deps.spatial?.onStatus?.(groupId, status);
  };

  const display = deps.spatial
    ? createSpatialDisplay(
        {
          onSettle: (owner) => {
            const slot = slotsByOwner.get(owner);
            if (slot) {
              deps.spatial?.onSettle?.(slot.groupId);
            }
          },
          onStatus: (owner) => {
            const slot = slotsByOwner.get(owner);
            if (slot) {
              reportStatus(slot.groupId);
            }
          },
          run: deps.spatial.run,
        },
        'group:'
      )
    : null;

  /** Releases a spatial slot's adjusted surface and abandons its work, keeping the raw composite. */
  const releaseSpatial = (slot: GroupSlot): void => {
    if (slot.spatial) {
      adjustBytes(-surfaceBytes(slot.spatial.adjusted));
      slot.spatial = null;
      display?.drop(slot.owner);
    }
  };

  /** Forgets slots already removed from their group's list, then reports each group's status. */
  const dropSlots = (slots: readonly GroupSlot[]): void => {
    adjustBytes(-slots.reduce((bytes, slot) => bytes + slotBytes(slot), 0));
    for (const slot of slots) {
      slotsByOwner.delete(slot.owner);
      if (frameSlots.get(slot.groupId) === slot) {
        frameSlots.delete(slot.groupId);
      }
      display?.drop(slot.owner);
    }
    for (const groupId of new Set(slots.map((slot) => slot.groupId))) {
      reportStatus(groupId);
    }
  };

  // A scope's OWN opacity/blend are applied by the consumer when the surface
  // lands, so they stay out of its shape key (an opacity scrub reuses the
  // surface); a CHILD scope's opacity/blend are baked in here, so children key
  // with them.
  //
  // A spatial scope's own stack applies to a separate adjusted surface, so its own key stays out too and a
  // parameter edit reuses the raw composite; a parent still keys a spatial child with it.
  const childScopeKey = (scope: GroupCompositeScope): string =>
    `${scopeShapeKey(scope, adjustmentsKey(scope.adjustments))}:${scope.opacity}:${scope.blendMode}`;
  const scopeShapeKey = (scope: GroupCompositeScope, ownKey: string): string =>
    `${scope.id}@${scope.start}-${scope.end}:${ownKey}(${scope.children.map(childScopeKey).join(',')})`;
  const scopeStructureKey = (scope: GroupCompositeScope): string =>
    `${scope.id}@${scope.start}-${scope.end}(${scope.children.map(scopeStructureKey).join(',')})`;

  /** Identity of what replaces or rides on a member's committed pixels: its filter preview and its float. */
  const contentKeyOf = (leafId: string, content: GroupSurfaceContent): string => {
    const preview = content.previews?.get(leafId);
    const float = content.float?.layerId === leafId ? content.float : null;
    const previewKey = preview ? `:p${objectId(preview.surface)}@${rectKey(preview.rect)}` : '';
    return float ? `${previewKey}:f${objectId(float.surface)}` : previewKey;
  };

  /** Everything a surface depends on except member pixels and float placement, which are tracked as damage. */
  const shapeKeyOf = (
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number,
    spatial: boolean
  ): string => {
    const memberKeys: string[] = [];
    for (let i = scope.start; i < scope.end; i += 1) {
      const leaf = members[i - baseIndex]!;
      if (content.excludeIds.has(leaf.id)) {
        memberKeys.push(`${leaf.id}:x`);
        continue;
      }
      const { layer } = leaf;
      const own = layer.type === 'raster' ? adjustmentsKey(layer.adjustments) : '-';
      const appearance = `${layer.opacity}:${layer.blendMode}:${matKey(memberMatrices[i - baseIndex]!)}:${own}`;
      memberKeys.push(`${leaf.id}:${appearance}${contentKeyOf(leaf.id, content)}`);
    }
    return `${scopeShapeKey(scope, spatial ? '~spatial' : adjustmentsKey(scope.adjustments))}|${memberKeys.join('|')}`;
  };

  /** A spatial scope's shape key without its members' and nested groups' looks, which {@link appearancesOf} tracks. */
  const structureKeyOf = (
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    content: GroupSurfaceContent,
    baseIndex: number
  ): string => {
    const memberKeys: string[] = [];
    for (let i = scope.start; i < scope.end; i += 1) {
      const { id } = members[i - baseIndex]!;
      memberKeys.push(content.excludeIds.has(id) ? `${id}:x` : `${id}${contentKeyOf(id, content)}`);
    }
    return `~spatial:${scopeStructureKey(scope)}|${memberKeys.join('|')}`;
  };

  /** The looks of a spatial scope's drawn members and nested groups, keyed by member id or `scope:<group id>`. */
  const appearancesOf = (
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    baseIndex: number,
    drawn: readonly DrawnEntry[]
  ): ReadonlyMap<string, DrawnAppearance> => {
    const appearances = new Map<string, DrawnAppearance>();
    const boundsById = new Map<string, Rect | null>();
    for (const { bounds, leaf, matrix } of drawn) {
      const { layer } = leaf;
      const adjustments = layer.type === 'raster' ? layer.adjustments : undefined;
      boundsById.set(leaf.id, bounds);
      appearances.set(leaf.id, {
        bounds,
        placement: `${layer.opacity}:${layer.blendMode}:${matKey(matrix)}`,
        spatial: hasSpatialAdjustments(adjustments),
        stack: adjustmentsKey(adjustments),
      });
    }
    const addScopes = (scopes: readonly GroupCompositeScope[]): void => {
      for (const child of scopes) {
        let bounds: Rect | null = null;
        for (let i = child.start; i < child.end; i += 1) {
          const leafBounds = boundsById.get(members[i - baseIndex]!.id);
          if (leafBounds) {
            bounds = bounds ? union(bounds, leafBounds) : leafBounds;
          }
        }
        appearances.set(`scope:${child.id}`, {
          bounds,
          placement: `${child.opacity}:${child.blendMode}`,
          spatial: hasSpatialAdjustments(child.adjustments),
          stack: adjustmentsKey(child.adjustments),
        });
        addScopes(child.children);
      }
    };
    addScopes(scope.children);
    return appearances;
  };

  /** Members whose committed cache pixels the surface shows, with their document-space bounds. */
  const drawnEntries = (
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number
  ): DrawnEntry[] => {
    const drawn: DrawnEntry[] = [];
    for (let i = scope.start; i < scope.end; i += 1) {
      const leaf = members[i - baseIndex]!;
      if (content.excludeIds.has(leaf.id) || leaf.layer.type !== 'raster') {
        continue;
      }
      const matrix = memberMatrices[i - baseIndex]!;
      const preview = content.previews?.get(leaf.id);
      const entry = deps.getCacheEntry(leaf.id) ?? null;
      const pixels = preview ? preview.rect : entry && !isEmpty(entry.rect) ? entry.rect : null;
      let bounds = pixels ? transformBounds(matrix, pixels) : null;
      const float = content.float?.layerId === leaf.id ? content.float : null;
      let drawnFloat: DrawnFloat | null = null;
      if (float && !isEmpty(float.rect)) {
        const placed = multiply(matrix, float.matrix);
        const landing = transformBounds(placed, float.rect);
        bounds = bounds ? union(bounds, landing) : landing;
        drawnFloat = {
          landing: transformBounds(placed, expand(float.rect, SOURCE_FILTER_RADIUS)),
          placement: `${rectKey(float.rect)}*${matKey(float.matrix)}`,
        };
      }
      drawn.push({ bounds, entry: preview ? null : entry, float: drawnFloat, leaf, matrix });
    }
    return drawn;
  };

  /**
   * Draws `[from, to)` (absolute plan indices, bottom first); `baseIndex` = absolute index of `members[0]`. Returns
   * how final the drawn members and nested groups are.
   */
  const drawRange = (
    ctx: RasterSurface['ctx'],
    view: Mat2d,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number,
    from: number,
    to: number,
    children: readonly GroupCompositeScope[],
    frame: boolean
  ): DisplayedPixelsState => {
    let inputs: DisplayedPixelsState = 'exact';
    let childIndex = 0;
    for (let i = from; i < to;) {
      const child = childIndex < children.length ? children[childIndex]! : null;
      if (child && i >= child.start && i < child.end) {
        const nested = resolve(child, members, memberMatrices, content, baseIndex, frame);
        if (nested) {
          inputs = worseDisplayedState(inputs, nested.state);
          ctx.save();
          ctx.globalAlpha = child.opacity;
          ctx.globalCompositeOperation = blendToComposite(child.blendMode);
          ctx.setTransform(view.a, view.b, view.c, view.d, view.e, view.f);
          ctx.drawImage(nested.surface.canvas, nested.rect.x, nested.rect.y);
          ctx.restore();
        }
        i = child.end;
        childIndex += 1;
        continue;
      }
      const leaf = members[i - baseIndex]!;
      const matrix = memberMatrices[i - baseIndex]!;
      i += 1;
      if (content.excludeIds.has(leaf.id) || leaf.layer.type !== 'raster') {
        continue;
      }
      const placed = multiply(view, matrix);
      const preview = content.previews?.get(leaf.id);
      const entry = preview ? undefined : deps.getCacheEntry(leaf.id);
      ctx.save();
      ctx.globalAlpha = leaf.layer.opacity;
      ctx.globalCompositeOperation = blendToComposite(leaf.layer.blendMode);
      ctx.setTransform(placed.a, placed.b, placed.c, placed.d, placed.e, placed.f);
      if (preview) {
        ctx.drawImage(preview.surface.canvas, preview.rect.x, preview.rect.y);
      } else if (entry && !isEmpty(entry.rect)) {
        const displayed = deps.getAdjustedSurface(leaf.layer, entry);
        inputs = worseDisplayedState(inputs, displayed.state);
        ctx.drawImage((displayed.surface ?? entry.surface).canvas, entry.rect.x, entry.rect.y);
      }
      const float = content.float?.layerId === leaf.id ? content.float : null;
      if (float) {
        const floated = multiply(placed, float.matrix);
        ctx.setTransform(floated.a, floated.b, floated.c, floated.d, floated.e, floated.f);
        ctx.drawImage(float.surface.canvas, float.rect.x, float.rect.y);
      }
      ctx.restore();
    }
    return inputs;
  };

  /**
   * Clears and redraws `region` (surface-local) of a slot, then applies the scope's own adjustments there unless they
   * are spatial (a spatial scope's surface stays raw). Records how final the drawn inputs were.
   */
  const paint = (
    slot: GroupSlot,
    region: Rect,
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number,
    spatial: boolean,
    frame: boolean
  ): void => {
    const ctx = slot.surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.beginPath();
    ctx.rect(region.x, region.y, region.width, region.height);
    ctx.clip();
    ctx.clearRect(region.x, region.y, region.width, region.height);
    const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: -slot.rect.x, f: -slot.rect.y };
    slot.inputs = drawRange(
      ctx,
      view,
      members,
      memberMatrices,
      content,
      baseIndex,
      scope.start,
      scope.end,
      scope.children,
      frame
    );
    ctx.restore();
    // Per-pixel adjustments are region independent and group stacks work in document pixels, so a region-limited
    // pass at the region's document origin equals a whole-surface one there.
    if (!spatial && !isIdentityAdjustments(scope.adjustments)) {
      const pixels = ctx.getImageData(region.x, region.y, region.width, region.height);
      applyAdjustments(pixels, scope.adjustments, { x: slot.rect.x + region.x, y: slot.rect.y + region.y });
      ctx.putImageData(pixels, region.x, region.y);
    }
  };

  /**
   * Document-space region the members changed since the slot was drawn; null when only a full redraw is safe. With
   * `appearances` (a spatial scope), changed looks add their old and new bounds. A spatial stack edited into another
   * spatial stack keeps its shown pixels until its result lands, so its bounds go to `reread` instead: redrawn so
   * the edit reaches its owner and its pending state is seen, but not damage for the adjusted copy.
   */
  const damagedRegion = (
    slot: GroupSlot,
    drawn: readonly DrawnEntry[],
    appearances: ReadonlyMap<string, DrawnAppearance> | null
  ): Damage | null => {
    if (drawn.filter((item) => item.entry).length !== slot.versions.size) {
      return null;
    }
    let region: Rect | null = null;
    let reread: Rect | null = null;
    if (appearances) {
      const before = slot.appearances;
      if (!before || before.size !== appearances.size) {
        return null;
      }
      for (const [id, next] of appearances) {
        const previous = before.get(id);
        if (!previous) {
          return null;
        }
        if (previous.placement === next.placement && previous.stack === next.stack) {
          continue;
        }
        const kept = previous.placement === next.placement && previous.spatial && next.spatial;
        for (const bounds of [previous.bounds, next.bounds]) {
          if (bounds && kept) {
            reread = reread ? union(reread, bounds) : bounds;
          } else if (bounds) {
            region = region ? union(region, bounds) : bounds;
          }
        }
      }
    }
    const float = drawn.find((item) => item.float)?.float ?? null;
    if (float?.placement !== slot.float?.placement) {
      for (const landing of [slot.float?.landing, float?.landing]) {
        if (landing) {
          region = region ? union(region, landing) : landing;
        }
      }
    }
    for (const { entry, leaf, matrix } of drawn) {
      if (!entry) {
        continue;
      }
      const drawnVersion = slot.versions.get(leaf.id);
      if (drawnVersion === undefined) {
        return null;
      }
      if (drawnVersion === entry.version) {
        continue;
      }
      const local = deps.damageSince(leaf.id, drawnVersion);
      if (!local) {
        return null;
      }
      const changed = transformBounds(
        matrix,
        expand(
          { height: local.height, width: local.width, x: entry.rect.x + local.x, y: entry.rect.y + local.y },
          SOURCE_FILTER_RADIUS
        )
      );
      region = region ? union(region, changed) : changed;
    }
    return { region, reread };
  };

  const versionsOf = (drawn: readonly DrawnEntry[]): ReadonlyMap<string, number> =>
    new Map(drawn.flatMap(({ entry, leaf }) => (entry ? [[leaf.id, entry.version] as const] : [])));

  /**
   * Shows a spatial scope's adjusted surface for the slot's raw composite: an interim now (`painted` is the raw
   * damage this read drew, `undefined` when none), the evaluated stack when its job lands. Jobs wait while an input
   * is still pending.
   */
  const showSpatial = (
    display: SpatialDisplay,
    slot: GroupSlot,
    scope: GroupCompositeScope,
    painted: Rect | null | undefined,
    published: boolean
  ): { readonly surface: RasterSurface; readonly own: DisplayedPixelsState } => {
    const raw = slot.surface;
    let spatial = slot.spatial;
    if (!spatial) {
      spatial = { adjusted: deps.createSurface(raw.width, raw.height), adjustedFor: null, rawVersion: 0, trail: [] };
      adjustBytes(surfaceBytes(spatial.adjusted));
      slot.spatial = spatial;
    }
    const state = spatial;
    if (painted !== undefined) {
      state.rawVersion += 1;
      state.trail.push({ damage: painted, first: state.rawVersion, version: state.rawVersion });
      if (state.trail.length > RAW_DAMAGE_TRAIL_LIMIT) {
        // Fold the oldest repaint into the next, so an adjusted copy from a long stroke ago still re-patches. An
        // unknown one already leaves every earlier version unknown: dropping it loses nothing.
        const oldest = state.trail.shift()!;
        const next = state.trail[0]!;
        if (oldest.damage) {
          state.trail[0] = {
            damage: next.damage && union(oldest.damage, next.damage),
            first: oldest.first,
            version: next.version,
          };
        }
      }
    }
    let reusable = state.adjustedFor !== null;
    if (state.adjusted.width !== raw.width || state.adjusted.height !== raw.height) {
      const before = surfaceBytes(state.adjusted);
      state.adjusted.resize(raw.width, raw.height);
      adjustBytes(surfaceBytes(state.adjusted) - before);
      reusable = false;
    }
    const key = adjustmentsKey(scope.adjustments);
    const version = state.rawVersion;
    const live = (): boolean => slotsByOwner.get(slot.owner) === slot && slot.spatial === state;
    const target: SpatialDisplayTarget = {
      adjustments: scope.adjustments,
      damageSince: (from) => rawDamageSince(state, from),
      hold: slot.inputs === 'pending' ? 'inputs' : published ? null : 'placeholder',
      key,
      liveVersion: () => (live() ? state.rawVersion : null),
      // Group stacks work in document pixels; the surface's (0,0) sits at the slot rect's origin.
      origin: { x: slot.rect.x, y: slot.rect.y },
      shown: () =>
        live() && state.adjustedFor?.version === version && state.adjustedFor.key === key ? state.adjusted : null,
      source: raw,
      version,
    };
    let rebuilt = false;
    if (state.adjustedFor?.version !== version || state.adjustedFor.key !== key) {
      rebuilt = display.paintInterim(slot.owner, state.adjusted, reusable, target);
      state.adjustedFor = { key, version };
    }
    display.request(slot.owner, target, rebuilt);
    return { own: display.state(slot.owner, target), surface: state.adjusted };
  };

  function resolve(
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number,
    frame: boolean
  ): GroupSurfaceResult | null {
    clock += 1;
    const drawn = drawnEntries(scope, members, memberMatrices, content, baseIndex);
    let bounds: Rect | null = null;
    for (const item of drawn) {
      if (item.bounds) {
        bounds = bounds ? union(bounds, item.bounds) : item.bounds;
      }
    }
    if (bounds === null) {
      // Keys fully determine validity, so warm slots stay for the other consumer.
      return null;
    }
    const rect = roundOut(bounds);
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }
    const spatial = display !== null && hasSpatialAdjustments(scope.adjustments);
    const shapeKey = shapeKeyOf(scope, members, memberMatrices, content, baseIndex, spatial);
    const structureKey = spatial ? structureKeyOf(scope, members, content, baseIndex) : null;
    const appearances = spatial ? appearancesOf(scope, members, baseIndex, drawn) : null;
    const slots = cache.get(scope.id) ?? [];
    let matching = slots.find((candidate) => candidate.shapeKey === shapeKey) ?? null;
    let damage: Damage | null | undefined;
    if (structureKey !== null && !(matching && sameRect(matching.rect, rect))) {
      // Only looks changed (a member's stack, opacity, blend or matrix, a nested group's stack, opacity or blend):
      // refresh the same-shaped slot needing the least repaint in place, so its adjusted copy keeps its pixels
      // outside the changed bounds. Frame and overview keys still settle into one slot each.
      let bestArea = Infinity;
      for (const candidate of slots) {
        if (candidate.structureKey !== structureKey || !sameRect(candidate.rect, rect)) {
          continue;
        }
        const candidateDamage = damagedRegion(candidate, drawn, appearances);
        const redrawn = candidateDamage && unionOf(candidateDamage.region, candidateDamage.reread);
        const area = !candidateDamage ? Infinity : redrawn ? redrawn.width * redrawn.height : 0;
        if (damage === undefined || area < bestArea) {
          matching = candidate;
          damage = candidateDamage;
          bestArea = area;
        }
      }
      if (damage !== undefined) {
        matching!.shapeKey = shapeKey;
      }
    }
    const whole: Rect = { height: rect.height, width: rect.width, x: 0, y: 0 };
    const toLocal = (region: Rect | null): Rect | null =>
      region &&
      intersect(roundOut(expand({ ...region, x: region.x - rect.x, y: region.y - rect.y }, ANTIALIAS_MARGIN)), whole);
    let slot: GroupSlot;
    /** Raw damage this read drew: a surface-local rect, null when unknown, undefined when nothing was drawn. */
    let painted: Rect | null | undefined;

    if (matching && sameRect(matching.rect, rect)) {
      slot = matching;
      damage = damage === undefined ? damagedRegion(slot, drawn, appearances) : damage;
      if (slot.stale) {
        // A member's displayed pixels changed without a version: redraw everything, but report only the content
        // damage, so a spatial scope keeps its adjusted pixels elsewhere until its own job lands.
        paint(slot, whole, scope, members, memberMatrices, content, baseIndex, spatial, frame);
        painted = damage ? (toLocal(damage.region) ?? NO_AREA) : null;
        deps.diagnostics?.increment('groupSurfaceRebuilds');
      } else {
        const region = damage ? toLocal(unionOf(damage.region, damage.reread)) : whole;
        if (region && !isEmpty(region)) {
          paint(slot, region, scope, members, memberMatrices, content, baseIndex, spatial, frame);
          painted = damage ? (toLocal(damage.region) ?? NO_AREA) : null;
          deps.diagnostics?.increment(damage ? 'groupSurfaceRefreshes' : 'groupSurfaceRebuilds');
        }
      }
    } else if (matching || slots.length >= SLOTS_PER_GROUP) {
      // New bounds or content reuse a slot's backing store, reallocating only when the size changed.
      slot = matching ?? slots[slots.length - 1]!;
      if (slot.surface.width !== rect.width || slot.surface.height !== rect.height) {
        const before = surfaceBytes(slot.surface);
        slot.surface.resize(rect.width, rect.height);
        adjustBytes(surfaceBytes(slot.surface) - before);
        deps.diagnostics?.increment('groupSurfaceAllocations');
      }
      slot.rect = rect;
      slot.shapeKey = shapeKey;
      paint(slot, whole, scope, members, memberMatrices, content, baseIndex, spatial, frame);
      painted = null;
      deps.diagnostics?.increment('groupSurfaceRebuilds');
    } else {
      slot = {
        float: null,
        groupId: scope.id,
        inputs: 'exact',
        lastUsed: clock,
        owner: `${scope.id}#${nextSlot++}`,
        rect,
        shapeKey,
        appearances: null,
        spatial: null,
        stale: false,
        structureKey: null,
        surface: deps.createSurface(rect.width, rect.height),
        versions: new Map(),
      };
      adjustBytes(surfaceBytes(slot.surface));
      slotsByOwner.set(slot.owner, slot);
      deps.diagnostics?.increment('groupSurfaceAllocations');
      paint(slot, whole, scope, members, memberMatrices, content, baseIndex, spatial, frame);
      painted = null;
      slots.unshift(slot);
      cache.set(scope.id, slots);
    }
    slot.versions = versionsOf(drawn);
    slot.stale = false;
    slot.structureKey = structureKey;
    slot.appearances = appearances;
    slot.float = drawn.find((item) => item.float)?.float ?? null;
    slot.lastUsed = clock;
    if (frame) {
      frameSlots.set(scope.id, slot);
    }
    const index = slots.indexOf(slot);
    if (index > 0) {
      slots.splice(index, 1);
      slots.unshift(slot);
    }
    let result: GroupSurfaceResult;
    if (spatial) {
      // A placeholder composite (no member has real pixels yet) is never worth a job.
      const published = drawn.some(
        (item) => item.float !== null || (item.entry ? item.entry.hasPublishedPixels : item.bounds !== null)
      );
      const shown = showSpatial(display!, slot, scope, painted, published);
      result = { rect: slot.rect, state: worseDisplayedState(slot.inputs, shown.own), surface: shown.surface };
    } else {
      releaseSpatial(slot);
      result = { rect: slot.rect, state: slot.inputs, surface: slot.surface };
    }
    reportStatus(scope.id);
    return result;
  }

  /** Removes `slots` from their group's list (and the group once empty), then forgets them. */
  const removeSlots = (groupId: string, removed: readonly GroupSlot[]): void => {
    const slots = cache.get(groupId);
    if (!slots) {
      return;
    }
    const kept = slots.filter((slot) => !removed.includes(slot));
    if (kept.length === 0) {
      cache.delete(groupId);
    } else {
      cache.set(groupId, kept);
    }
    dropSlots(removed);
  };

  return {
    byteSize: () => totalBytes,
    cancelSpatial: () => {
      for (const owner of display?.cancel() ?? []) {
        const spatial = slotsByOwner.get(owner)?.spatial;
        if (spatial) {
          spatial.adjustedFor = null;
        }
      }
    },
    clear: () => {
      const all = [...cache.values()].flat();
      cache.clear();
      dropSlots(all);
    },
    evict: (budgetBytes, sinceTick) => {
      if (totalBytes <= budgetBytes) {
        return 0;
      }
      const candidates = [...cache.entries()]
        .flatMap(([groupId, slots]) => slots.map((slot) => ({ groupId, slot })))
        .filter(({ slot }) => slot.lastUsed <= sinceTick)
        .sort((a, b) => a.slot.lastUsed - b.slot.lastUsed);
      let evicted = 0;
      for (const { groupId, slot } of candidates) {
        if (totalBytes <= budgetBytes) {
          break;
        }
        removeSlots(groupId, [slot]);
        evicted += 1;
      }
      return evicted;
    },
    get: (scope, members, memberMatrices, content, options) =>
      resolve(scope, members, memberMatrices, content, scope.start, options?.frame ?? false),
    invalidate: (groupIds) => {
      for (const groupId of groupIds) {
        for (const slot of cache.get(groupId) ?? []) {
          slot.stale = true;
        }
      }
    },
    prune: (liveGroupIds) => {
      for (const [id, slots] of cache) {
        if (!liveGroupIds.has(id)) {
          removeSlots(id, slots);
        }
      }
    },
    retainSpatial: (keep) => {
      for (const [id, slots] of cache) {
        const spatialSlots = slots.filter((slot) => slot.spatial);
        if (spatialSlots.length > 0 && !keep(id)) {
          removeSlots(id, spatialSlots);
        }
      }
    },
    tick: () => clock,
  };
};
