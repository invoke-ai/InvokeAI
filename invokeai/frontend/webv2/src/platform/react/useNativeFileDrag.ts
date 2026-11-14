import type { DragEvent as ReactDragEvent } from 'react';

import { useCallback, useRef, useState } from 'react';

import { isEventFromOwnDom } from './isEventFromOwnDom';
import { useMountEffect } from './useMountEffect';

/** Files dragged in from outside the page over a drop surface: their MIME types and the zone under the pointer. */
export interface NativeFileDrag<Zone extends string> {
  /** One per dragged file, readable while it hovers (contents are not); empty when the OS gives no type. */
  fileTypes: readonly string[];
  zone: Zone | null;
}

interface DragPoint {
  clientX: number;
  clientY: number;
}

// A file drag over the surface's own elements. React also bubbles drag events from portalled content a descendant
// opened, such as a confirmation dialog or a menu: that content is not the surface wherever it is painted, so the
// surface neither tracks nor takes such a drag. Leaving the surface for it ends the drag (see `onDragLeave`), and
// entering the surface from it starts a new one, so skipping its events keeps enters and leaves balanced.
const isFileDragOnSurface = (event: ReactDragEvent): boolean =>
  isEventFromOwnDom(event) && event.dataTransfer.types.includes('Files');

const fileTypesOf = (event: ReactDragEvent): string[] =>
  [...event.dataTransfer.items].filter((item) => item.kind === 'file').map((item) => item.type);

const isWithin = (event: ReactDragEvent<HTMLElement>): boolean =>
  event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget);

/**
 * Adapts HTML5 file drags from the operating system, which pointer-sensor drag libraries never see, for one drop
 * surface: spread `handlers` on its root, render zones from `drag` that `findZone` locates, and receive a drop on a
 * zone in `onDrop`. Every file drag over the surface's own elements is taken, so the browser never opens the file in
 * place of the app; where no zone is under the pointer the drag shows that it cannot drop. Portalled content the
 * surface's descendants open (a dialog, a menu) is not the surface: a drag over it is left to the page's navigation
 * guard, so a drop on an open confirmation never lands behind it.
 *
 * The drag ends when it leaves the surface for the rest of the page, leaves the window or is cancelled (a leave with
 * no related target), and on any drop or `dragend`. It is not tied to window focus: a window manager that moves focus
 * with the pointer would otherwise end a drag still in progress.
 */
export const useNativeFileDrag = <Zone extends string>({
  findZone,
  onDrop,
}: {
  /** The zone under the point within the surface's root, or null where none takes files. */
  findZone: (root: HTMLElement, point: DragPoint) => Zone | null;
  onDrop: (files: File[], zone: Zone) => void;
}) => {
  const [drag, setDrag] = useState<NativeFileDrag<Zone> | null>(null);
  // The zone the last dragenter or dragover found, as the browser's drop effect reported it. The drop reads this rather
  // than the zones, which an update flushed between the drop's listeners may already have taken down.
  const zoneRef = useRef<Zone | null>(null);
  const pendingEndRef = useRef<number | null>(null);
  const cancelPendingEnd = useCallback(() => {
    if (pendingEndRef.current !== null) {
      window.clearTimeout(pendingEndRef.current);
      pendingEndRef.current = null;
    }
  }, []);
  const end = useCallback(() => {
    cancelPendingEnd();
    zoneRef.current = null;
    setDrag(null);
  }, [cancelPendingEnd]);

  useMountEffect(() => {
    // Bubble phase: a drop on the surface is handled by the surface first; these end a drag that finished elsewhere.
    window.addEventListener('drop', end);
    window.addEventListener('dragend', end);
    return () => {
      window.removeEventListener('drop', end);
      window.removeEventListener('dragend', end);
      cancelPendingEnd();
    };
  });

  const track = useCallback(
    (event: ReactDragEvent<HTMLElement>, entering: boolean) => {
      cancelPendingEnd();
      const zone = findZone(event.currentTarget, event);
      zoneRef.current = zone;
      setDrag((current) =>
        // A drag coming in from outside the surface is a new one: its files' types are read afresh.
        current === null || entering
          ? { fileTypes: fileTypesOf(event), zone }
          : current.zone === zone
            ? current
            : { ...current, zone }
      );
      return zone;
    },
    [cancelPendingEnd, findZone]
  );

  const onDragEnter = useCallback(
    (event: ReactDragEvent<HTMLElement>) => {
      if (!isFileDragOnSurface(event)) {
        return;
      }
      event.preventDefault();
      track(event, !isWithin(event));
    },
    [track]
  );
  const onDragOver = useCallback(
    (event: ReactDragEvent<HTMLElement>) => {
      if (!isFileDragOnSurface(event)) {
        return;
      }
      event.preventDefault();
      event.dataTransfer.dropEffect = track(event, false) === null ? 'none' : 'copy';
    },
    [track]
  );
  const onDragLeave = useCallback(
    (event: ReactDragEvent<HTMLElement>) => {
      if (!isFileDragOnSurface(event) || isWithin(event)) {
        return;
      }
      if (event.relatedTarget === null) {
        // Leaving the window or a cancelled drag; some engines also report a move between the surface's own elements
        // this way, and a dragenter or dragover on the surface follows such a move in the same task, keeping the drag.
        cancelPendingEnd();
        pendingEndRef.current = window.setTimeout(end, 0);
        return;
      }
      end();
    },
    [cancelPendingEnd, end]
  );
  const handleDrop = useCallback(
    (event: ReactDragEvent<HTMLElement>) => {
      if (!isFileDragOnSurface(event)) {
        return;
      }
      event.preventDefault();
      const zone = zoneRef.current;
      const files = [...event.dataTransfer.files];
      end();
      if (zone !== null) {
        onDrop(files, zone);
      }
    },
    [end, onDrop]
  );

  return { drag, handlers: { onDragEnter, onDragLeave, onDragOver, onDrop: handleDrop } };
};
