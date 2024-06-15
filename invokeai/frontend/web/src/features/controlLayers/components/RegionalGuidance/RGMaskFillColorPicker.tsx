import { Flex, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import RgbColorPicker from 'common/components/RgbColorPicker';
import { stopPropagation } from 'common/util/stopPropagation';
import { rgbColorToString } from 'features/canvas/util/colorToString';
import { rgFillChanged, selectRGOrThrow } from 'features/controlLayers/store/regionalGuidanceSlice';
import { memo, useCallback } from 'react';
import type { RgbColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';

type Props = {
  id: string;
};

export const RGMaskFillColorPicker = memo(({ id }: Props) => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const fill = useAppSelector((s) => selectRGOrThrow(s.regionalGuidance, id).fill);
  const onChange = useCallback(
    (fill: RgbColor) => {
      dispatch(rgFillChanged({ id, fill }));
    },
    [dispatch, id]
  );
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Flex
          as="button"
          aria-label={t('controlLayers.maskPreviewColor')}
          borderRadius="full"
          borderWidth={1}
          bg={rgbColorToString(fill)}
          w={8}
          h={8}
          cursor="pointer"
          tabIndex={-1}
          onDoubleClick={stopPropagation} // double click expands the layer
        />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody minH={64}>
          <RgbColorPicker color={fill} onChange={onChange} withNumberInput />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

RGMaskFillColorPicker.displayName = 'RGMaskFillColorPicker';
