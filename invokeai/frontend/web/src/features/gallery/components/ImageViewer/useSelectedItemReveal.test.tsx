// @vitest-environment happy-dom
/**
 * Mounted-DOM coverage for the preview components' reveal wiring. The machine's sequencing has its
 * own unit tests; what only a real mount can verify is the wiring the components rely on — effect
 * lifecycles, StrictMode double-invocation, the image <-> video component swap over the one shared
 * machine, and media readiness driven by a real <video> element's event.
 */
import {
  $gallerySelection,
  markNextSelectionAutoSwitched,
  recordGallerySelection,
  resetGallerySelectionSource,
} from 'features/gallery/store/gallerySelectionSource';
import { atom } from 'nanostores';
import { act, StrictMode } from 'react';
import type { Root } from 'react-dom/client';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { SelectedItemRevealMachine } from './selectedItemReveal';
import { createSelectedItemRevealMachine } from './selectedItemReveal';
import { usePaintedItemName, useSelectedItemReveal } from './useSelectedItemReveal';

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean;
}
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

const DURATION_MS = 2000;
const MEDIA_GRACE_MS = 1000;

/** What the viewer context provides: one machine writing one shared flag. */
const createHarness = () => {
  const $revealed = atom(false);
  const machine = createSelectedItemRevealMachine({
    setRevealed: (revealed) => $revealed.set(revealed),
    durationMs: DURATION_MS,
    mediaGraceMs: MEDIA_GRACE_MS,
  });
  return { machine, isRevealed: () => $revealed.get() };
};

const RevealProbe = ({
  machine,
  itemName,
  isMediaReady = true,
  hasProgressImage = true,
  isProgressImageResolving = false,
}: {
  machine: SelectedItemRevealMachine;
  itemName: string | null;
  isMediaReady?: boolean;
  hasProgressImage?: boolean;
  isProgressImageResolving?: boolean;
}) => {
  useSelectedItemReveal({
    revealMachine: machine,
    renderedItemName: itemName,
    isMediaReady,
    shouldShowProgressInViewer: true,
    hasProgressImage,
    isProgressImageResolving,
  });
  return null;
};

/** CurrentVideoPreview's readiness wiring: a real <video> element feeding usePaintedItemName. */
const VideoProbe = ({ machine, videoName }: { machine: SelectedItemRevealMachine; videoName: string | null }) => {
  const { isMediaReady, onPainted } = usePaintedItemName(videoName);
  useSelectedItemReveal({
    revealMachine: machine,
    renderedItemName: videoName,
    isMediaReady,
    shouldShowProgressInViewer: true,
    hasProgressImage: true,
    isProgressImageResolving: false,
  });
  return <video data-testid="probe-video" key={videoName ?? undefined} onLoadedData={onPainted} />;
};

describe('useSelectedItemReveal (mounted)', () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    vi.useFakeTimers();
    resetGallerySelectionSource();
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
    vi.useRealTimers();
  });

  /** The user picks an item: publish the selection the way the store listener would. */
  const select = (itemName: string) => {
    act(() => {
      recordGallerySelection(itemName);
    });
  };

  it('reveals a mid-render click and lowers when the duration elapses', () => {
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    select('b.png');
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="b.png" />);
    });
    expect(h.isRevealed()).toBe(true);

    act(() => {
      vi.advanceTimersByTime(DURATION_MS);
    });
    expect(h.isRevealed()).toBe(false);
  });

  it('survives StrictMode double-invoked effects', () => {
    const h = createHarness();
    act(() => {
      root.render(
        <StrictMode>
          <RevealProbe machine={h.machine} itemName="a.png" />
        </StrictMode>
      );
    });
    select('b.png');
    act(() => {
      root.render(
        <StrictMode>
          <RevealProbe machine={h.machine} itemName="b.png" />
        </StrictMode>
      );
    });
    expect(h.isRevealed(), 'the reveal survives the doubled effects').toBe(true);
  });

  it('keeps one reveal running across an image -> video component swap mid-reveal', () => {
    // One machine serves both components, so the swap must neither kill the running reveal nor
    // restart its clock.
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    select('b.png');
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="b.png" />);
    });
    expect(h.isRevealed()).toBe(true);
    act(() => {
      vi.advanceTimersByTime(500);
    });

    // The other preview component takes over rendering the same item.
    act(() => {
      root.render(<RevealProbe key="video-side" machine={h.machine} itemName="b.png" />);
    });
    expect(h.isRevealed(), 'the reveal survives the swap').toBe(true);

    act(() => {
      vi.advanceTimersByTime(DURATION_MS - 500);
    });
    expect(h.isRevealed(), 'and still ends on the original clock').toBe(false);
  });

  it('holds a video click until a real loadeddata event, then reveals', () => {
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    select('b.mp4');
    act(() => {
      root.render(<VideoProbe machine={h.machine} videoName="b.mp4" />);
    });
    expect(h.isRevealed(), 'held for the unpainted video').toBe(false);

    const video = container.querySelector('video');
    expect(video).not.toBeNull();
    act(() => {
      video?.dispatchEvent(new Event('loadeddata'));
    });
    expect(h.isRevealed(), 'revealed once the frame is there').toBe(true);
  });

  it('reveals after the media grace even if loadeddata never fires', () => {
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    select('never-loads.mp4');
    act(() => {
      root.render(<VideoProbe machine={h.machine} videoName="never-loads.mp4" />);
    });
    expect(h.isRevealed()).toBe(false);
    act(() => {
      vi.advanceTimersByTime(MEDIA_GRACE_MS);
    });
    expect(h.isRevealed(), 'the click still lands').toBe(true);
  });

  it('resets readiness when the video element is swapped for another video', () => {
    // A stale "painted" from the previous element must not lift the overlay onto the new, black
    // one — readiness is compared by name, so the swap cannot inherit it.
    const h = createHarness();
    select('a.mp4');
    act(() => {
      root.render(<VideoProbe machine={h.machine} videoName="a.mp4" />);
    });
    act(() => {
      container.querySelector('video')?.dispatchEvent(new Event('loadeddata'));
    });
    act(() => {
      vi.advanceTimersByTime(DURATION_MS);
    });

    select('b.mp4');
    act(() => {
      root.render(<VideoProbe machine={h.machine} videoName="b.mp4" />);
    });
    expect(h.isRevealed(), 'held: the new element has not painted').toBe(false);
    act(() => {
      container.querySelector('video')?.dispatchEvent(new Event('loadeddata'));
    });
    expect(h.isRevealed()).toBe(true);
  });

  it('does not reveal an auto-switched selection', () => {
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    act(() => {
      markNextSelectionAutoSwitched();
      recordGallerySelection('finished.png');
    });
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="finished.png" />);
    });
    expect(h.isRevealed()).toBe(false);
  });

  it('settles a selection made while neither preview was mounted, instead of replaying it', () => {
    // Comparison mode unmounts both previews while the provider (and its machine) stays alive; the
    // provider settles selections through noteSelection. Returning must not fire a reveal for a
    // click that was already visible when it happened.
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    act(() => {
      root.unmount();
    });
    root = createRoot(container);

    act(() => {
      recordGallerySelection('picked-while-comparing.png');
      h.machine.noteSelection($gallerySelection.get());
    });

    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="picked-while-comparing.png" />);
    });
    expect(h.isRevealed()).toBe(false);
  });

  it('lowers everything on provider teardown, with no timer left behind', () => {
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    select('b.png');
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="b.png" />);
    });
    expect(h.isRevealed()).toBe(true);

    act(() => {
      h.machine.reset();
    });
    expect(h.isRevealed()).toBe(false);
    act(() => {
      vi.advanceTimersByTime(DURATION_MS * 2);
    });
    expect(h.isRevealed()).toBe(false);
  });

  it('reveals a repeat click on the item already on screen', () => {
    // Re-picking the displayed item changes no state and no rendered name — only the selection
    // generation moves. This is the click the previous name-comparison design could never see.
    const h = createHarness();
    select('a.png');
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    act(() => {
      vi.advanceTimersByTime(DURATION_MS);
    });
    expect(h.isRevealed()).toBe(false);

    select('a.png');
    expect(h.isRevealed(), 'the repeat click is a new selection and reveals').toBe(true);
  });

  it('keeps the reveal when the provider settles selections while a preview is mounted', () => {
    // The provider's subscription calls noteSelection on every selection; while a preview is
    // attached that must be a no-op, or every selection would be settled before the component's
    // own sync can classify it — and no click would ever reveal.
    const h = createHarness();
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="a.png" />);
    });
    act(() => {
      recordGallerySelection('b.png');
      h.machine.noteSelection($gallerySelection.get());
    });
    act(() => {
      root.render(<RevealProbe machine={h.machine} itemName="b.png" />);
    });
    expect(h.isRevealed(), 'attach makes the provider settle a no-op while mounted').toBe(true);
  });
});
