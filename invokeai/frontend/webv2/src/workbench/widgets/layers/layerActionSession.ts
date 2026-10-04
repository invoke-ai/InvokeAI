export interface LayerActionRequest {
  token: number;
  signal: AbortSignal;
}

export interface LayerActionSession {
  begin(): LayerActionRequest | null;
  cancel(): void;
  finish(token: number): void;
  isCurrent(token: number): boolean;
}

export const createLayerActionSession = (): LayerActionSession => {
  let nextToken = 0;
  let active: { controller: AbortController; token: number } | null = null;

  return {
    begin: () => {
      if (active) {
        return null;
      }
      const controller = new AbortController();
      const token = ++nextToken;
      active = { controller, token };
      return { signal: controller.signal, token };
    },
    cancel: () => {
      nextToken += 1;
      active?.controller.abort();
      active = null;
    },
    finish: (token) => {
      if (active?.token === token) {
        active = null;
      }
    },
    isCurrent: (token) => active?.token === token && !active.controller.signal.aborted,
  };
};
