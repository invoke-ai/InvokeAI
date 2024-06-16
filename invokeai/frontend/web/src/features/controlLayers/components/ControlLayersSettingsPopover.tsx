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
import { MaskOpacity } from 'features/controlLayers/components/MaskOpacity';
import { invertScrollChanged } from 'features/controlLayers/store/canvasV2Slice';
import type { ChangeEvent } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { RiSettings4Fill } from 'react-icons/ri';

const ControlLayersSettingsPopover = () => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const invertScroll = useAppSelector((s) => s.canvasV2.tool.invertScroll);
  const onChangeInvertScroll = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => dispatch(invertScrollChanged(e.target.checked)),
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
            <MaskOpacity />
            <FormControl w="full">
              <FormLabel flexGrow={1}>{t('unifiedCanvas.invertBrushSizeScrollDirection')}</FormLabel>
              <Checkbox isChecked={invertScroll} onChange={onChangeInvertScroll} />
            </FormControl>
          </Flex>
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};

export default memo(ControlLayersSettingsPopover);
