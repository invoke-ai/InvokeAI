import { Box, Flex, FormControl, FormLabel } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import RgbaColorPicker from 'common/components/ColorPicker/RgbaColorPicker';
import {
  selectInfillColorValue,
  selectInfillMethod,
  setInfillColorValue,
  useParamsDispatch,
} from 'features/controlLayers/store/paramsSlice';
import { memo, useCallback } from 'react';
import type { RgbaColor } from 'react-colorful';
import { useTranslation } from 'react-i18next';

const ParamInfillColorOptions = () => {
  const dispatchParams = useParamsDispatch();

  const infillColor = useAppSelector(selectInfillColorValue);
  const infillMethod = useAppSelector(selectInfillMethod);

  const { t } = useTranslation();

  const handleInfillColor = useCallback(
    (v: RgbaColor) => {
      dispatchParams(setInfillColorValue, v);
    },
    [dispatchParams]
  );

  return (
    <Flex flexDir="column" gap={4}>
      <FormControl isDisabled={infillMethod !== 'color'}>
        <FormLabel>{t('parameters.infillColorValue')}</FormLabel>
        <Box w="full" pt={2} pb={2}>
          <RgbaColorPicker color={infillColor} onChange={handleInfillColor} />
        </Box>
      </FormControl>
    </Flex>
  );
};

export default memo(ParamInfillColorOptions);
