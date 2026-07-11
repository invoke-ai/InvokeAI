import type { KeyboardSensorOptions } from '@dnd-kit/core';

import { sortableKeyboardCoordinates } from '@dnd-kit/sortable';

export const LAYER_KEYBOARD_SENSOR_OPTIONS = {
  coordinateGetter: sortableKeyboardCoordinates,
  keyboardCodes: {
    cancel: ['Escape'],
    end: ['Enter'],
    start: ['Enter'],
  },
} satisfies KeyboardSensorOptions;
