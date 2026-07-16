import type { LayerExportGuard } from '@workbench/canvas-engine/api';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect } from '@workbench/canvas-engine/types';

export interface StagedPreviewState {
  surface: RasterSurface;
  width: number;
  height: number;
  placement?: Rect & { opacity: number };
}

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
}

/** Owns render-preview surfaces and monotonic async publication guards. */
export class PreviewStateController {
  private staged: StagedPreviewState | null = null;
  private stagedToken = 0;
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

  hasFilter(layerId: string): boolean {
    return this.filters.has(layerId);
  }
  getFilter(layerId: string): FilterPreviewState | undefined {
    return this.filters.get(layerId);
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
    this.clearFilters();
    this.sam = null;
  }
}
