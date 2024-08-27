import type { SerializableObject } from 'common/types';
import type { CanvasManager } from 'features/controlLayers/konva/CanvasManager';
import type { Logger } from 'roarr';

export abstract class CanvasModuleABC {
  abstract id: string;
  abstract type: string;
  abstract path: string[];
  abstract manager: CanvasManager;
  abstract log: Logger;
  abstract subscriptions: Set<() => void>;

  abstract getLoggingContext: () => SerializableObject;
  abstract destroy: () => void;
  abstract repr: () => SerializableObject & {
    id: string;
    path: string[];
    type: string;
  };
}
