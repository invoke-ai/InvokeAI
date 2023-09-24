import { Box, Flex, FormControl, FormLabel } from '@chakra-ui/react';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSeamlessXAxis from './ParamSeamlessXAxis';
import ParamSeamlessYAxis from './ParamSeamlessYAxis';

const ParamSeamless = () => {
  const { t } = useTranslation();

  const isSeamlessEnabled = useFeatureStatus('seamless').isFeatureEnabled;

  if (!isSeamlessEnabled) {
    return null;
  }

  return (
    <FormControl>
      <FormLabel>{t('parameters.seamlessTiling')}</FormLabel>{' '}
      <Flex sx={{ gap: 5 }}>
        <Box flexGrow={1}>
          <ParamSeamlessXAxis />
        </Box>
        <Box flexGrow={1}>
          <ParamSeamlessYAxis />
        </Box>
      </Flex>
    </FormControl>
  );
};

export default memo(ParamSeamless);
