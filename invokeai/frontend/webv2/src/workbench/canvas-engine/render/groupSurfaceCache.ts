/**
 * Document-space group composites keyed by scope shape, member appearance, effective matrices and in-scope
 * previews. Own opacity/blend apply at draw time. A stable slot refreshes in place: only the region its members
 * damaged since the last draw is cleared and redrawn, and geometry changes redraw without reallocating when the
 * size holds. Document-resolution buffers resample transformed members twice and upscale above 100% zoom.
 */

import type { CanvasRasterLayerContractV2 } from '@workbench/canvas-engine/contracts';
import type { CanvasDiagnostics } from '@workbench/canvas-engine/diagnostics';
import type { SemanticLeaf } from '@workbench/canvas-engine/document-model/semanticLeaf';
import type { LayerCacheEntry } from '@workbench/canvas-engine/render/layerCache';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect } from '@workbench/canvas-engine/types';

import { multiply } from '@workbench/canvas-engine/math/mat2d';
import { expand, intersect, isEmpty, roundOut, transformBounds, union } from '@workbench/canvas-engine/math/rect';
import { adjustmentsKey, applyAdjustments, isIdentityAdjustments } from '@workbench/canvas-engine/render/adjustments';
import { blendToComposite } from '@workbench/canvas-engine/render/compositor';

import type { GroupCompositeScope, GroupSurfaceContent } from './groupCompositeScopes';

export interface GroupSurfaceResult {
  readonly surface: RasterSurface;
  /** Document-space placement of the surface. */
  readonly rect: Rect;
}

export interface GroupSurfaceDeps {
  createSurface(width: number, height: number): RasterSurface;
  /** Receives the running byte total after every allocation, resize and release. */
  onBytesChange?(bytes: number): void;
  getCacheEntry(layerId: string): LayerCacheEntry | undefined;
  /** Surface-local damage a member's cache received since `version`, or null when unknown. */
  damageSince(layerId: string, version: number): Rect | null;
  /** The member's own adjusted pixels (its personal stack), or null for raw. */
  getAdjustedSurface(layer: CanvasRasterLayerContractV2, entry: LayerCacheEntry): RasterSurface | null;
  diagnostics?: CanvasDiagnostics;
}

export interface GroupSurfaceCache {
  /** Total RGBA bytes the cached group surfaces hold, for the memory budget. */
  byteSize(): number;
  get(
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent
  ): GroupSurfaceResult | null;
  /** Drops every cached group not named; call when document structure changes. */
  prune(liveGroupIds: ReadonlySet<string>): void;
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

interface GroupSlot {
  shapeKey: string;
  rect: Rect;
  readonly surface: RasterSurface;
  /** Cache version of every member pixel source the surface currently shows. */
  versions: ReadonlyMap<string, number>;
  /** Placement of the floating selection the surface shows, so a drag refreshes in place. */
  float: DrawnFloat | null;
  lastUsed: number;
}

interface DrawnFloat {
  readonly placement: string;
  /** Document-space bounds the float's resampled pixels can reach. */
  readonly landing: Rect;
}

interface DrawnEntry {
  readonly leaf: SemanticLeaf;
  readonly matrix: Mat2d;
  readonly entry: LayerCacheEntry | null;
  readonly bounds: Rect | null;
  readonly float: DrawnFloat | null;
}

const surfaceBytes = (surface: RasterSurface): number => surface.width * surface.height * 4;

export const createGroupSurfaceCache = (deps: GroupSurfaceDeps): GroupSurfaceCache => {
  const cache = new Map<string, GroupSlot[]>();
  const objectIds = new WeakMap<object, number>();
  let nextObjectId = 1;
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

  const dropSlots = (slots: readonly GroupSlot[]): void => {
    adjustBytes(-slots.reduce((bytes, slot) => bytes + surfaceBytes(slot.surface), 0));
  };

  // A scope's OWN opacity/blend are applied by the consumer when the surface
  // lands, so they stay out of its shape key (an opacity scrub reuses the
  // surface); a CHILD scope's opacity/blend are baked in here, so children key
  // with them.
  const childScopeKey = (scope: GroupCompositeScope): string =>
    `${scopeShapeKey(scope)}:${scope.opacity}:${scope.blendMode}`;
  const scopeShapeKey = (scope: GroupCompositeScope): string =>
    `${scope.id}@${scope.start}-${scope.end}:${adjustmentsKey(scope.adjustments)}(${scope.children
      .map(childScopeKey)
      .join(',')})`;

  /** Everything a surface depends on except member pixels and float placement, which are tracked as damage. */
  const shapeKeyOf = (
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number
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
      const preview = content.previews?.get(leaf.id);
      const float = content.float?.layerId === leaf.id ? content.float : null;
      const previewKey = preview ? `:p${objectId(preview.surface)}@${rectKey(preview.rect)}` : '';
      const floatKey = float ? `:f${objectId(float.surface)}` : '';
      const appearance = `${layer.opacity}:${layer.blendMode}:${matKey(memberMatrices[i - baseIndex]!)}:${own}`;
      memberKeys.push(`${leaf.id}:${appearance}${previewKey}${floatKey}`);
    }
    return `${scopeShapeKey(scope)}|${memberKeys.join('|')}`;
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

  /** Draws `[from, to)` (absolute plan indices, bottom first); `baseIndex` = absolute index of `members[0]`. */
  const drawRange = (
    ctx: RasterSurface['ctx'],
    view: Mat2d,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number,
    from: number,
    to: number,
    children: readonly GroupCompositeScope[]
  ): void => {
    let childIndex = 0;
    for (let i = from; i < to;) {
      const child = childIndex < children.length ? children[childIndex]! : null;
      if (child && i >= child.start && i < child.end) {
        const nested = resolve(child, members, memberMatrices, content, baseIndex);
        if (nested) {
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
        const adjusted = deps.getAdjustedSurface(leaf.layer, entry);
        ctx.drawImage((adjusted ?? entry.surface).canvas, entry.rect.x, entry.rect.y);
      }
      const float = content.float?.layerId === leaf.id ? content.float : null;
      if (float) {
        const floated = multiply(placed, float.matrix);
        ctx.setTransform(floated.a, floated.b, floated.c, floated.d, floated.e, floated.f);
        ctx.drawImage(float.surface.canvas, float.rect.x, float.rect.y);
      }
      ctx.restore();
    }
  };

  /** Clears and redraws `region` (surface-local) of a slot, then applies the scope's own adjustments there. */
  const paint = (
    slot: GroupSlot,
    region: Rect,
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number
  ): void => {
    const ctx = slot.surface.ctx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.beginPath();
    ctx.rect(region.x, region.y, region.width, region.height);
    ctx.clip();
    ctx.clearRect(region.x, region.y, region.width, region.height);
    const view: Mat2d = { a: 1, b: 0, c: 0, d: 1, e: -slot.rect.x, f: -slot.rect.y };
    drawRange(ctx, view, members, memberMatrices, content, baseIndex, scope.start, scope.end, scope.children);
    ctx.restore();
    // Every adjustment is per-pixel, so a region-limited pass equals a whole-surface one there.
    if (!isIdentityAdjustments(scope.adjustments)) {
      const pixels = ctx.getImageData(region.x, region.y, region.width, region.height);
      applyAdjustments(pixels, scope.adjustments);
      ctx.putImageData(pixels, region.x, region.y);
    }
  };

  /** Document-space region the members changed since the slot was drawn; null when only a full redraw is safe. */
  const damagedRegion = (slot: GroupSlot, drawn: readonly DrawnEntry[]): { region: Rect | null } | null => {
    if (drawn.filter((item) => item.entry).length !== slot.versions.size) {
      return null;
    }
    let region: Rect | null = null;
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
    return { region };
  };

  const versionsOf = (drawn: readonly DrawnEntry[]): ReadonlyMap<string, number> =>
    new Map(drawn.flatMap(({ entry, leaf }) => (entry ? [[leaf.id, entry.version] as const] : [])));

  function resolve(
    scope: GroupCompositeScope,
    members: readonly SemanticLeaf[],
    memberMatrices: readonly Mat2d[],
    content: GroupSurfaceContent,
    baseIndex: number
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
    const shapeKey = shapeKeyOf(scope, members, memberMatrices, content, baseIndex);
    const slots = cache.get(scope.id) ?? [];
    const matching = slots.find((candidate) => candidate.shapeKey === shapeKey) ?? null;
    const whole: Rect = { height: rect.height, width: rect.width, x: 0, y: 0 };
    let slot: GroupSlot;

    if (matching && sameRect(matching.rect, rect)) {
      slot = matching;
      const damage = damagedRegion(slot, drawn);
      const region = damage
        ? damage.region &&
          intersect(
            roundOut(
              expand({ ...damage.region, x: damage.region.x - rect.x, y: damage.region.y - rect.y }, ANTIALIAS_MARGIN)
            ),
            whole
          )
        : whole;
      if (region && !isEmpty(region)) {
        paint(slot, region, scope, members, memberMatrices, content, baseIndex);
        deps.diagnostics?.increment(damage ? 'groupSurfaceRefreshes' : 'groupSurfaceRebuilds');
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
      paint(slot, whole, scope, members, memberMatrices, content, baseIndex);
      deps.diagnostics?.increment('groupSurfaceRebuilds');
    } else {
      slot = {
        lastUsed: clock,
        rect,
        shapeKey,
        float: null,
        surface: deps.createSurface(rect.width, rect.height),
        versions: new Map(),
      };
      adjustBytes(surfaceBytes(slot.surface));
      deps.diagnostics?.increment('groupSurfaceAllocations');
      paint(slot, whole, scope, members, memberMatrices, content, baseIndex);
      slots.unshift(slot);
      cache.set(scope.id, slots);
    }
    slot.versions = versionsOf(drawn);
    slot.float = drawn.find((item) => item.float)?.float ?? null;
    slot.lastUsed = clock;
    const index = slots.indexOf(slot);
    if (index > 0) {
      slots.splice(index, 1);
      slots.unshift(slot);
    }
    return { rect: slot.rect, surface: slot.surface };
  }

  return {
    byteSize: () => totalBytes,
    clear: () => {
      for (const slots of cache.values()) {
        dropSlots(slots);
      }
      cache.clear();
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
        const slots = cache.get(groupId)!;
        slots.splice(slots.indexOf(slot), 1);
        if (slots.length === 0) {
          cache.delete(groupId);
        }
        dropSlots([slot]);
        evicted += 1;
      }
      return evicted;
    },
    get: (scope, members, memberMatrices, content) => resolve(scope, members, memberMatrices, content, scope.start),
    prune: (liveGroupIds) => {
      for (const [id, slots] of cache) {
        if (!liveGroupIds.has(id)) {
          dropSlots(slots);
          cache.delete(id);
        }
      }
    },
    tick: () => clock,
  };
};
