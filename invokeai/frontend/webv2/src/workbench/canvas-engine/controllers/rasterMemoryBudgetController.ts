import { DEFAULT_CACHE_BUDGET_BYTES } from '@workbench/canvas-engine/render/layerCache';

export type RasterBackgroundPurpose =
  | 'background-snapshot'
  | 'invocation-composite'
  | 'layer-operation'
  | 'thumbnail'
  | 'raster-export'
  | 'psd-export';

export interface RasterMemoryLease {
  release(): void;
}

export type RasterMemoryReservationResult =
  | { status: 'ok'; lease: RasterMemoryLease }
  | { status: 'over-budget'; requestedBytes: number; availableBytes: number };

export interface RasterMemorySnapshot {
  baseBytes: number;
  derivedBytes: number;
  groupBytes: number;
  decodedBytes: number;
  detachedBytes: number;
  reservedBytes: number;
  totalBytes: number;
  /** Bytes held above the budget because the working set requires them. */
  overageBytes: number;
}

/** Allocation classes whose owners push their running totals at every allocation, resize and release. */
export type RasterMemoryCategory = 'base' | 'derived' | 'group' | 'decoded';

interface GenerationLease extends RasterMemoryLease {
  readonly generation: number;
}

const normalizedBytes = (bytes: number): number => Math.max(0, Math.ceil(bytes));

/** Owns byte accounting, reservations, and the pins that keep in-use caches resident. */
export class RasterMemoryBudgetController {
  readonly budgetBytes: number;
  private readonly categoryBytes: Record<RasterMemoryCategory, number> = { base: 0, decoded: 0, derived: 0, group: 0 };
  private detachedBytes = 0;
  private reservedBytes = 0;
  private readonly generationLeases = new Map<number, Set<GenerationLease>>();
  private readonly pins = new Map<string, Set<RasterMemoryLease>>();
  private disposed = false;

  constructor(options: { budgetBytes?: number } = {}) {
    this.budgetBytes = normalizedBytes(options.budgetBytes ?? DEFAULT_CACHE_BUDGET_BYTES);
  }

  /** Records an owner's current total for one allocation class. */
  setCategoryBytes(category: RasterMemoryCategory, bytes: number): void {
    this.categoryBytes[category] = normalizedBytes(bytes);
  }

  /** Accounts caller-owned detached pixels until the returned lease is released. */
  trackDetached(bytes: number): RasterMemoryLease {
    if (this.disposed) {
      return { release: () => undefined };
    }
    const trackedBytes = normalizedBytes(bytes);
    this.detachedBytes += trackedBytes;
    return this.createOwnedLease(() => {
      this.detachedBytes = Math.max(0, this.detachedBytes - trackedBytes);
    });
  }

  /** Reserves bytes for cancellable background preparation; lifecycle transitions release the generation. */
  reserve(
    requestedBytes: number,
    options: { generation: number; purpose: RasterBackgroundPurpose }
  ): RasterMemoryReservationResult {
    const bytes = normalizedBytes(requestedBytes);
    const availableBytes = this.getAvailableBytes();
    if (this.disposed || bytes > availableBytes) {
      return { availableBytes, requestedBytes: bytes, status: 'over-budget' };
    }
    this.reservedBytes += bytes;
    const lease = this.createGenerationLease(options.generation, () => {
      this.reservedBytes = Math.max(0, this.reservedBytes - bytes);
    });
    return { lease, status: 'ok' };
  }

  /** Reserves bytes for an in-flight operation independently of lifecycle generations. */
  reserveOperation(
    requestedBytes: number,
    _options: { purpose: RasterBackgroundPurpose }
  ): RasterMemoryReservationResult {
    const bytes = normalizedBytes(requestedBytes);
    const availableBytes = this.getAvailableBytes();
    if (this.disposed || bytes > availableBytes) {
      return { availableBytes, requestedBytes: bytes, status: 'over-budget' };
    }
    this.reservedBytes += bytes;
    return {
      lease: this.createOwnedLease(() => {
        this.reservedBytes = Math.max(0, this.reservedBytes - bytes);
      }),
      status: 'ok',
    };
  }

  getAvailableBytes(): number {
    return Math.max(0, this.budgetBytes - this.snapshot().totalBytes);
  }

  /** Keeps a layer's cache resident (no eviction or trim) until the owning operation releases it. */
  pin(layerId: string): RasterMemoryLease {
    if (this.disposed) {
      return { release: () => undefined };
    }
    const lease = this.createOwnedLease(() => {
      const layerPins = this.pins.get(layerId);
      layerPins?.delete(lease);
      if (layerPins?.size === 0) {
        this.pins.delete(layerId);
      }
    });
    const layerPins = this.pins.get(layerId) ?? new Set<RasterMemoryLease>();
    layerPins.add(lease);
    this.pins.set(layerId, layerPins);
    return lease;
  }

  isPinned(layerId: string): boolean {
    return (this.pins.get(layerId)?.size ?? 0) > 0;
  }

  releaseGeneration(generation: number): void {
    const leases = [...(this.generationLeases.get(generation) ?? [])];
    for (const lease of leases) {
      lease.release();
    }
  }

  snapshot(): RasterMemorySnapshot {
    const { base, decoded, derived, group } = this.categoryBytes;
    const totalBytes = base + derived + group + decoded + this.detachedBytes + this.reservedBytes;
    return {
      baseBytes: base,
      decodedBytes: decoded,
      derivedBytes: derived,
      detachedBytes: this.detachedBytes,
      groupBytes: group,
      overageBytes: Math.max(0, totalBytes - this.budgetBytes),
      reservedBytes: this.reservedBytes,
      totalBytes,
    };
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    for (const generation of this.generationLeases.keys()) {
      this.releaseGeneration(generation);
    }
    this.pins.clear();
    this.reservedBytes = 0;
    this.detachedBytes = 0;
  }

  private createGenerationLease(generation: number, onRelease: () => void): GenerationLease {
    let released = false;
    const lease: GenerationLease = {
      generation,
      release: () => {
        if (released) {
          return;
        }
        released = true;
        onRelease();
        const generationSet = this.generationLeases.get(generation);
        generationSet?.delete(lease);
        if (generationSet?.size === 0) {
          this.generationLeases.delete(generation);
        }
      },
    };
    const generationSet = this.generationLeases.get(generation) ?? new Set<GenerationLease>();
    generationSet.add(lease);
    this.generationLeases.set(generation, generationSet);
    return lease;
  }

  private createOwnedLease(onRelease: () => void): RasterMemoryLease {
    let released = false;
    return {
      release: () => {
        if (released) {
          return;
        }
        released = true;
        onRelease();
      },
    };
  }
}
