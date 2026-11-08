export class PaletteDebouncer {
  #timer: number | null = null;
  readonly #commit: (value: string) => void;
  readonly #delayMs: number;

  constructor(delayMs: number, commit: (value: string) => void) {
    this.#commit = commit;
    this.#delayMs = delayMs;
  }

  cancel = (): void => {
    if (this.#timer !== null) {
      window.clearTimeout(this.#timer);
      this.#timer = null;
    }
  };

  commit = (value: string): void => {
    this.cancel();
    this.#commit(value);
  };

  schedule = (value: string): void => {
    this.cancel();
    this.#timer = window.setTimeout(() => {
      this.#timer = null;
      this.#commit(value);
    }, this.#delayMs);
  };
}
