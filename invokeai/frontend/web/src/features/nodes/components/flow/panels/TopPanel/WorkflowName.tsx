import { Text } from '@chakra-ui/layout';
import { useAppSelector } from 'app/store/storeHooks';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const TopCenterPanel = () => {
  const { t } = useTranslation();
  const name = useAppSelector((state) => state.workflow.name);
  const isTouched = useAppSelector((state) => state.workflow.isTouched);

  return (
    <Text
      m={2}
      fontSize="lg"
      userSelect="none"
      noOfLines={1}
      wordBreak="break-all"
      fontWeight={600}
      opacity={0.8}
    >
      {name || t('workflows.unnamedWorkflow')}
      {isTouched ? ` (${t('common.unsaved')})` : ''}
    </Text>
  );
};

export default memo(TopCenterPanel);
