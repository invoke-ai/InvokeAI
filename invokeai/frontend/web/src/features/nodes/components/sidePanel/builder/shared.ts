import { useAppSelector } from 'app/store/storeHooks';
import { getPrefixedId } from 'features/controlLayers/konva/util';
import { selectFormRootElementId } from 'features/nodes/store/workflowSlice';
import { useMemo } from 'react';

export const EDIT_MODE_WRAPPER_CLASS_NAME = getPrefixedId('edit-mode-wrapper', '-');

export const getEditModeWrapperId = (id: string) => `${id}-edit-mode-wrapper`;

export const useIsRootElement = (id: string) => {
  const rootElementId = useAppSelector(selectFormRootElementId);
  const isRootElement = useMemo(() => rootElementId === id, [rootElementId, id]);
  return isRootElement;
};
