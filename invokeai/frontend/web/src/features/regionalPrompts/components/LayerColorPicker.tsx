import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { useLayer } from 'features/regionalPrompts/hooks/layerStateHooks';
import { promptRegionLayerColorChanged } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { useCallback } from 'react';
import type { RgbColor } from 'react-colorful';
import { PiEyedropperBold } from 'react-icons/pi';

type Props = {
  id: string;
};

export const LayerColorPicker = ({ id }: Props) => {
  const layer = useLayer(id);
  const dispatch = useAppDispatch();
  const onColorChange = useCallback(
    (color: RgbColor) => {
      dispatch(promptRegionLayerColorChanged({ layerId: id, color }));
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
          <RgbColorPicker color={layer.color} onChange={onColorChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
