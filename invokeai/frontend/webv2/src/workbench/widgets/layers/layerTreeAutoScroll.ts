/** The band at each edge of the list where a drag scrolls it, and the fastest it scrolls. */
export const AUTO_SCROLL_ZONE_PX = 36;
export const AUTO_SCROLL_MAX_PX_PER_FRAME = 14;

export interface LayerTreeAutoScroller {
  /** Reads the pointer's viewport y and scrolls while it rests in an edge band; stops the moment it leaves. */
  update(clientY: number): void;
  stop(): void;
}

/**
 * Deterministic edge scrolling for drags: velocity grows linearly with depth into the band and the
 * loop runs only while there is velocity, so a pointer that leaves the band stops scrolling on the
 * next frame and a finished drag stops it at once.
 */
export const createLayerTreeAutoScroller = (
  element: { readonly current: HTMLElement | null },
  schedule: (callback: () => void) => number = (callback) => requestAnimationFrame(callback),
  cancel: (handle: number) => void = (handle) => cancelAnimationFrame(handle)
): LayerTreeAutoScroller => {
  let frame: number | null = null;
  let velocity = 0;
  let carry = 0;
  const stop = (): void => {
    if (frame !== null) {
      cancel(frame);
    }
    frame = null;
    velocity = 0;
    carry = 0;
  };
  // Sub-pixel velocity accumulates until it moves a whole pixel; a list that cannot move any further
  // parks the loop, and the next pointer movement restarts it.
  const step = (): void => {
    const host = element.current;
    if (!host || velocity === 0) {
      frame = null;
      return;
    }
    const total = velocity + carry;
    const whole = Math.trunc(total);
    carry = total - whole;
    if (whole !== 0) {
      const before = host.scrollTop;
      host.scrollTop = before + whole;
      if (host.scrollTop === before) {
        frame = null;
        return;
      }
    }
    frame = schedule(step);
  };
  return {
    stop,
    update: (clientY) => {
      const host = element.current;
      if (!host) {
        return;
      }
      const rect = host.getBoundingClientRect();
      const fromTop = clientY - rect.top;
      const fromBottom = rect.bottom - clientY;
      let next = 0;
      if (fromTop < AUTO_SCROLL_ZONE_PX) {
        next = -AUTO_SCROLL_MAX_PX_PER_FRAME * (1 - Math.max(0, fromTop) / AUTO_SCROLL_ZONE_PX);
      } else if (fromBottom < AUTO_SCROLL_ZONE_PX) {
        next = AUTO_SCROLL_MAX_PX_PER_FRAME * (1 - Math.max(0, fromBottom) / AUTO_SCROLL_ZONE_PX);
      }
      if (next === 0) {
        stop();
        return;
      }
      velocity = next;
      if (frame === null) {
        frame = schedule(step);
      }
    },
  };
};
