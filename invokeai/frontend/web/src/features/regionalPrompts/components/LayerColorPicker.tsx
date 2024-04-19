import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import {
  isRegionalPromptLayer,
  rpLayerColorChanged,
  selectRegionalPromptsSlice,
} from 'features/regionalPrompts/store/regionalPromptsSlice';
import { memo, useCallback, useMemo } from 'react';
import type { RgbColor } from 'react-colorful';
import { PiEyedropperBold } from 'react-icons/pi';
import { assert } from 'tsafe';

type Props = {
  id: string;
};

export const LayerColorPicker = memo(({ id }: Props) => {
  const selectColor = useMemo(
    () =>
      createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) => {
        const layer = regionalPrompts.present.layers.find((l) => l.id === id);
        assert(isRegionalPromptLayer(layer), `Layer ${id} not found or not an RP layer`);
        return layer.color;
      }),
    [id]
  );
  const color = useAppSelector(selectColor);
  const dispatch = useAppDispatch();
  const onColorChange = useCallback(
    (color: RgbColor) => {
      dispatch(rpLayerColorChanged({ layerId: id, color }));
    },
    [dispatch, id]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label="color picker" size="sm" borderRadius="base" icon={<PiEyedropperBold />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <RgbColorPicker color={color} onChange={onColorChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

LayerColorPicker.displayName = 'LayerColorPicker';
