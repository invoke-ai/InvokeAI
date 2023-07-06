import { Flex } from '@chakra-ui/react';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamBoundingBoxHeight from './ParamBoundingBoxHeight';
import ParamBoundingBoxWidth from './ParamBoundingBoxWidth';

const ParamBoundingBoxCollapse = () => {
  const { t } = useTranslation();

  return (
    <IAICollapse label={t('parameters.boundingBoxHeader')}>
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamBoundingBoxWidth />
        <ParamBoundingBoxHeight />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamBoundingBoxCollapse);
