/**
 * A Map that allows for subscribing to changes and getting a snapshot of the current state.
 *
 * It can be used with the `useSyncExternalStore` hook to sync the state of the map with a React component.
 *
 * Reactivity is shallow, so changes to nested objects will not trigger a re-render.
 */
export class SyncableMap<K, V> extends Map<K, V> {
  private subscriptions = new Set<() => void>();
  private lastSnapshot: Map<K, V> | null = null;

  constructor(entries?: readonly (readonly [K, V])[] | null) {
    super(entries);
  }

  set = (key: K, value: V): this => {
    super.set(key, value);
    this.notifySubscribers();
    return this;
  };

  delete = (key: K): boolean => {
    const result = super.delete(key);
    this.notifySubscribers();
    return result;
  };

  clear = (): void => {
    super.clear();
    this.notifySubscribers();
  };

  /**
   * Notify all subscribers that the map has changed.
   */
  private notifySubscribers = () => {
    for (const callback of this.subscriptions) {
      callback();
    }
  };

  /**
   * Subscribe to changes to the map.
   * @param callback A function to call when the map changes
   * @returns A function to unsubscribe from changes
   */
  subscribe = (callback: () => void): (() => void) => {
    this.subscriptions.add(callback);
    return () => {
      this.subscriptions.delete(callback);
    };
  };

  /**
   * Get a snapshot of the current state of the map.
   * @returns A snapshot of the current state of the map
   */
  getSnapshot = (): Map<K, V> => {
    const currentSnapshot = new Map(this);
    if (!this.lastSnapshot || !this.areSnapshotsEqual(this.lastSnapshot, currentSnapshot)) {
      this.lastSnapshot = currentSnapshot;
    }

    return this.lastSnapshot;
  };

  /**
   * Compare two snapshots to determine if they are equal.
   * @param snapshotA The first snapshot to compare
   * @param snapshotB The second snapshot to compare
   * @returns Whether the two snapshots are equal
   */
  private areSnapshotsEqual = (snapshotA: Map<K, V>, snapshotB: Map<K, V>): boolean => {
    if (snapshotA.size !== snapshotB.size) {
      return false;
    }

    for (const [key, value] of snapshotA) {
      if (!Object.is(value, snapshotB.get(key))) {
        return false;
      }
    }

    return true;
  };
}
