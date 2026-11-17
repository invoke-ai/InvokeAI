import type { DragEvent as ReactDragEvent } from 'react';

import { useCallback, useRef, useState } from 'react';

import { isEventFromOwnDom } from './isEventFromOwnDom';
import { useMountEffect } from './useMountEffect';

export interface NativeFileDrag<Zone extends string> {
  /** One per dragged file, readable while it hovers (contents are not); empty when the OS gives no type. */
  fileTypes: readonly string[];
  zone: Zone | null;
}

interface DragPoint {
  clientX: number;
  clientY: number;
}

// Portalled content a descendant opened is not the surface; skipping its events keeps enters and leaves balanced.
const isFileDragOnSurface = (event: ReactDragEvent): boolean =>
  isEventFromOwnDom(event) && event.dataTransfer.types.includes('Files');

const fileTypesOf = (event: ReactDragEvent): string[] =>
  [...event.dataTransfer.items].filter((item) => item.kind === 'file').map((item) => item.type);

const isWithin = (event: ReactDragEvent<HTMLElement>): boolean =>
  event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget);

/**
 * OS file drags, which pointer-sensor libraries never see. Every drag over the surface is taken so the browser never
 * opens the file; it ends on leaving the surface or window, a drop or `dragend`, never on focus changes.
 */
export const useNativeFileDrag = <Zone extends string>({
  findZone,
  onDrop,
}: {
  findZone: (root: HTMLElement, point: DragPoint) => Zone | null;
  onDrop: (files: File[], zone: Zone) => void;
}) => {
  const [drag, setDrag] = useState<NativeFileDrag<Zone> | null>(null);
  // The drop reads this, not the zones, which an update between the drop's listeners may have taken down.
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
        // Some engines report a move between the surface's own elements this way; a following enter keeps the drag.
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
