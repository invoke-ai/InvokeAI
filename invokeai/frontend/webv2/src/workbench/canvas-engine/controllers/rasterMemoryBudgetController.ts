import { DEFAULT_CACHE_BUDGET_BYTES } from '@workbench/canvas-engine/render/layerCache';

export type RasterBackgroundPurpose =
  | 'background-snapshot'
  | 'invocation-composite'
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
  decodedBytes: number;
  detachedBytes: number;
  reservedBytes: number;
  totalBytes: number;
}

interface GenerationLease extends RasterMemoryLease {
  readonly generation: number;
}

const normalizedBytes = (bytes: number): number => Math.max(0, Math.ceil(bytes));

/** Owns byte accounting and generation-scoped background allocation leases. */
export class RasterMemoryBudgetController {
  readonly budgetBytes: number;
  private baseBytes = 0;
  private derivedBytes = 0;
  private decodedBytes = 0;
  private reportedDetachedBytes = 0;
  private leasedDetachedBytes = 0;
  private reservedBytes = 0;
  private readonly generationLeases = new Map<number, Set<GenerationLease>>();
  private readonly pins = new Map<string, Set<RasterMemoryLease>>();
  private disposed = false;

  constructor(options: { budgetBytes?: number } = {}) {
    this.budgetBytes = normalizedBytes(options.budgetBytes ?? DEFAULT_CACHE_BUDGET_BYTES);
  }

  setBaseBytes(bytes: number): void {
    this.baseBytes = normalizedBytes(bytes);
  }

  setDerivedBytes(bytes: number): void {
    this.derivedBytes = normalizedBytes(bytes);
  }

  setDecodedBytes(bytes: number): void {
    this.decodedBytes = normalizedBytes(bytes);
  }

  setDetachedBytes(bytes: number): void {
    this.reportedDetachedBytes = normalizedBytes(bytes);
  }

  trackDetached(bytes: number, _generation: number): RasterMemoryLease {
    if (this.disposed) {
      return { release: () => undefined };
    }
    const trackedBytes = normalizedBytes(bytes);
    this.leasedDetachedBytes += trackedBytes;
    let released = false;
    return {
      release: () => {
        if (released) {
          return;
        }
        released = true;
        this.leasedDetachedBytes = Math.max(0, this.leasedDetachedBytes - trackedBytes);
      },
    };
  }

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

  pin(layerId: string, generation: number): RasterMemoryLease {
    if (this.disposed) {
      return { release: () => undefined };
    }
    const lease = this.createGenerationLease(generation, () => {
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

  /** Pins a cache for an in-flight operation independently of lifecycle generations. */
  pinOperation(layerId: string): RasterMemoryLease {
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

  pinnedLayerIds(): string[] {
    return [...this.pins.keys()];
  }

  releaseGeneration(generation: number): void {
    const leases = [...(this.generationLeases.get(generation) ?? [])];
    for (const lease of leases) {
      lease.release();
    }
  }

  snapshot(): RasterMemorySnapshot {
    const detachedBytes = this.reportedDetachedBytes + this.leasedDetachedBytes;
    return {
      baseBytes: this.baseBytes,
      decodedBytes: this.decodedBytes,
      derivedBytes: this.derivedBytes,
      detachedBytes,
      reservedBytes: this.reservedBytes,
      totalBytes: this.baseBytes + this.derivedBytes + this.decodedBytes + detachedBytes + this.reservedBytes,
    };
  }

  dispose(): void {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    while (this.generationLeases.size > 0) {
      const generation = this.generationLeases.keys().next().value;
      if (generation === undefined) {
        break;
      }
      this.releaseGeneration(generation);
    }
    this.pins.clear();
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
