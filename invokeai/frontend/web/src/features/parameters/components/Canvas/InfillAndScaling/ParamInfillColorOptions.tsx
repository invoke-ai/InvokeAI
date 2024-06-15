import { Box, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIColorPicker from 'common/components/IAIColorPicker';
import { setInfillColorValue } from 'features/controlLayers/store/canvasV2Slice';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';

const ParamInfillColorOptions = () => {
  const dispatch = useAppDispatch();

  const infillColor = useAppSelector((s) => s.canvasV2.compositing.infillColorValue);
  const infillMethod = useAppSelector((s) => s.canvasV2.compositing.infillMethod);

  const { t } = useTranslation();

  const handleInfillColor = useCallback(
    (v: RgbaColor) => {
      dispatch(setInfillColorValue(v));
    },
    [dispatch]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl isDisabled={infillMethod !== 'color'}>
        <FormLabel>{t('parameters.infillColorValue')}</FormLabel>
        <Box w="full" pt={2} pb={2}>
          <IAIColorPicker color={infillColor} onChange={handleInfillColor} />
        </Box>
      </FormControl>
    </Flex>
  );
};

export default memo(ParamInfillColorOptions);
