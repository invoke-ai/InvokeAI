import {
  Checkbox,
  Flex,
  FormControl,
  FormLabel,
  IconButton,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { setShouldInvertBrushSizeScrollDirection } from 'features/canvas/store/canvasSlice';
import { GlobalMaskLayerOpacity } from 'features/controlLayers/components/GlobalMaskLayerOpacity';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

const ControlLayersSettingsPopover = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const shouldInvertBrushSizeScrollDirection = useAppSelector((s) => s.canvas.shouldInvertBrushSizeScrollDirection);
  const handleChangeShouldInvertBrushSizeScrollDirection = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(setShouldInvertBrushSizeScrollDirection(e.target.checked)),
    [dispatch]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <IconButton aria-label={t('common.settingsLabel')} icon={<RiSettings4Fill />} />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <Flex direction="column" gap={2}>
            <GlobalMaskLayerOpacity />
            <FormControl w="full">
              <FormLabel flexGrow={1}>{t('unifiedCanvas.invertBrushSizeScrollDirection')}</FormLabel>
              <Checkbox
                isChecked={shouldInvertBrushSizeScrollDirection}
                onChange={handleChangeShouldInvertBrushSizeScrollDirection}
              />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ControlLayersSettingsPopover);
