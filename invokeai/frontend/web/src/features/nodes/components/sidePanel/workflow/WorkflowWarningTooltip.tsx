import { Flex, Text } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';

export const WorkflowWarningTooltip = () => {
  const { t } = useTranslation();
  return (
    <Flex flexDir="column" gap="2">
      <Text fontWeight="semibold">{t('toast.loadedWithWarnings')}</Text>
      <Flex flexDir="column">
        <Text>{t('common.toResolve')}:</Text>
        <Text>
          {t('nodes.editMode')} &gt;&gt; {t('nodes.updateAllNodes')} &gt;&gt; {t('common.save')}
        </Text>
      </Flex>
    </Flex>
  );
};
