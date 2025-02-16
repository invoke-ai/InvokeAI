import { getPrefixedId } from 'features/controlLayers/konva/util';

export const EDIT_MODE_WRAPPER_CLASS_NAME = getPrefixedId('edit-mode-wrapper', '-');

export const getEditModeWrapperId = (id: string) => `${id}-edit-mode-wrapper`;
