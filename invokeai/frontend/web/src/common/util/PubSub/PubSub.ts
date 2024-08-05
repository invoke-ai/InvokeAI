export type Listener<T> = (newValue: T, oldValue: T) => void;
export type CompareFn<T> = (a: T, b: T) => boolean;

/**
 * A simple PubSub implementation.
 *
 * @template T The type of the value to be published.
 * @param initialValue The initial value to publish.
 */
export class PubSub<T> {
  private _listeners: Set<Listener<T>> = new Set();
  private _oldValue: T;
  private _compareFn: CompareFn<T>;

  public constructor(initialValue: T, compareFn?: CompareFn<T>) {
    this._oldValue = initialValue;
    this._compareFn = compareFn || ((a, b) => a === b);
  }

  /**
   * Subscribes to the PubSub.
   * @param listener The listener to be called when the value is published.
   * @returns A function that can be called to unsubscribe the listener.
   */
  public subscribe = (listener: Listener<T>): (() => void) => {
    this._listeners.add(listener);

    return () => {
      this.unsubscribe(listener);
    };
  };

  /**
   * Unsubscribes a listener from the PubSub.
   * @param listener The listener to unsubscribe.
   */
  public unsubscribe = (listener: Listener<T>): void => {
    this._listeners.delete(listener);
  };

  /**
   * Publishes a new value to the PubSub.
   * @param newValue The new value to publish.
   */
  public publish = (newValue: T): void => {
    if (!this._compareFn(this._oldValue, newValue)) {
      for (const listener of this._listeners) {
        listener(newValue, this._oldValue);
      }
      this._oldValue = newValue;
    }
  };

  /**
   * Clears all listeners from the PubSub.
   */
  public off = (): void => {
    this._listeners.clear();
  };

  /**
   * Gets the current value of the PubSub.
   * @returns The current value of the PubSub.
   */
  public getValue = (): T | undefined => {
    return this._oldValue;
  };

  /**
   * Gets the listeners of the PubSub.
   * @returns The listeners of the PubSub.
   */
  public getListeners = (): Set<Listener<T>> => {
    return this._listeners;
  };
}
