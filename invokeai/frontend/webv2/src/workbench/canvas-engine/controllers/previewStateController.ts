import type { LayerExportGuard } from '@workbench/canvas-engine/capabilities';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

export interface StagedPreviewState {
  surface: RasterSurface;
  width: number;
  height: number;
  /** Absolute document rect pinned at submission, never the live bbox. */
  placement: Rect & { opacity: number };
}

/** Where one pending result lands, outlined by the overlay. */
export interface StagedRegionState {
  readonly rect: Rect;
  readonly selected: boolean;
}

const normalizeRegions = (regions: readonly StagedRegionState[]): StagedRegionState[] => {
  const unique = new Map<string, StagedRegionState>();
  for (const { rect, selected } of regions) {
    const key = `${rect.x}:${rect.y}:${rect.width}:${rect.height}`;
    unique.set(key, { rect: { ...rect }, selected: selected || unique.get(key)?.selected === true });
  }
  return [...unique.values()].sort((left, right) => Number(left.selected) - Number(right.selected));
};

const sameRegions = (left: readonly StagedRegionState[], right: readonly StagedRegionState[]): boolean =>
  left.length === right.length &&
  left.every((region, index) => {
    const other = right[index]!;
    return (
      region.selected === other.selected &&
      region.rect.x === other.rect.x &&
      region.rect.y === other.rect.y &&
      region.rect.width === other.rect.width &&
      region.rect.height === other.rect.height
    );
  });

export interface FilterPreviewState {
  surface: RasterSurface;
  rect: Rect;
  guard: LayerExportGuard;
}

export interface SamPreviewState {
  data: RasterSurface;
  rect: Rect;
  guard: LayerExportGuard;
  isolated: boolean;
  /** The mask's true-edge outline in document space, traced once at publish. */
  outline?: Path2D | null;
}

/** Owns render-preview surfaces and monotonic async publication guards. */
export class PreviewStateController {
  private staged: StagedPreviewState | null = null;
  private stagedToken = 0;
  private stagedRegions: readonly StagedRegionState[] = [];
  private readonly filters = new Map<string, FilterPreviewState>();
  private readonly filterTokens = new Map<string, number>();
  private readonly guardedFilterTokens = new Map<string, number>();
  private sam: SamPreviewState | null = null;

  getStaged(): StagedPreviewState | null {
    return this.staged;
  }
  nextStagedToken(): number {
    return ++this.stagedToken;
  }
  isStagedTokenCurrent(token: number): boolean {
    return token === this.stagedToken;
  }
  publishStaged(token: number, preview: StagedPreviewState): boolean {
    if (!this.isStagedTokenCurrent(token)) {
      return false;
    }
    this.staged = preview;
    return true;
  }
  clearStaged(): boolean {
    this.stagedToken += 1;
    const hadPreview = this.staged !== null;
    this.staged = null;
    return hadPreview;
  }

  getStagedRegions(): readonly StagedRegionState[] {
    return this.stagedRegions;
  }
  /**
   * Replaces the pending-result outlines, each rect once (results of one run share it; selected when any of them
   * is) with the selected one last so an overlap never hides it. False when unchanged: nothing needs repainting.
   */
  setStagedRegions(regions: readonly StagedRegionState[]): boolean {
    const next = normalizeRegions(regions);
    if (sameRegions(this.stagedRegions, next)) {
      return false;
    }
    this.stagedRegions = next;
    return true;
  }

  hasFilter(layerId: string): boolean {
    return this.filters.has(layerId);
  }
  getFilter(layerId: string): FilterPreviewState | undefined {
    return this.filters.get(layerId);
  }
  /** How many layers carry a filter preview, so a caller can skip snapshotting when none do. */
  filterCount(): number {
    return this.filters.size;
  }
  filterSnapshot(): Map<string, FilterPreviewState> {
    return new Map(this.filters);
  }
  filterLayerIds(): string[] {
    return [...new Set([...this.filters.keys(), ...this.filterTokens.keys(), ...this.guardedFilterTokens.keys()])];
  }
  clearFilter(layerId: string): boolean {
    this.filterTokens.set(layerId, (this.filterTokens.get(layerId) ?? 0) + 1);
    this.guardedFilterTokens.delete(layerId);
    return this.filters.delete(layerId);
  }
  beginGuardedFilter(layerId: string): number {
    const token = (this.filterTokens.get(layerId) ?? 0) + 1;
    this.filterTokens.set(layerId, token);
    this.guardedFilterTokens.set(layerId, token);
    return token;
  }
  finishGuardedFilter(layerId: string, token: number): void {
    if (this.guardedFilterTokens.get(layerId) === token) {
      this.guardedFilterTokens.delete(layerId);
    }
  }
  isFilterTokenCurrent(layerId: string, token: number): boolean {
    return this.filterTokens.get(layerId) === token;
  }
  publishFilter(layerId: string, token: number, preview: FilterPreviewState): boolean {
    if (!this.isFilterTokenCurrent(layerId, token)) {
      return false;
    }
    this.filters.set(layerId, preview);
    return true;
  }
  clearFilters(): void {
    for (const layerId of this.filterLayerIds()) {
      this.clearFilter(layerId);
    }
  }

  getSam(): SamPreviewState | null {
    return this.sam;
  }
  setSam(preview: SamPreviewState): SamPreviewState | null {
    const previous = this.sam;
    this.sam = preview;
    return previous;
  }
  clearSam(): SamPreviewState | null {
    const previous = this.sam;
    this.sam = null;
    return previous;
  }

  dispose(): void {
    this.clearStaged();
    this.stagedRegions = [];
    this.clearFilters();
    this.sam = null;
  }
}
