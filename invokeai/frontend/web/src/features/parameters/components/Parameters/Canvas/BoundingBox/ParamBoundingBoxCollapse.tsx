import { Flex, useDisclosure } from '@chakra-ui/react';
import { useTranslation } from 'react-i18next';
import IAICollapse from 'common/components/IAICollapse';
import { memo } from 'react';
import ParamBoundingBoxWidth from './ParamBoundingBoxWidth';
import ParamBoundingBoxHeight from './ParamBoundingBoxHeight';

const ParamBoundingBoxCollapse = () => {
  const { t } = useTranslation();
  const { isOpen, onToggle } = useDisclosure();

  return (
    <IAICollapse
      label={t('parameters.boundingBoxHeader')}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <Flex sx={{ gap: 2, flexDirection: 'column' }}>
        <ParamBoundingBoxWidth />
        <ParamBoundingBoxHeight />
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamBoundingBoxCollapse);
