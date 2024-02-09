import { Flex, Text } from '@invoke-ai/ui-library';
import { useTranslation } from 'react-i18next';
import { useGetNodesNeedUpdate } from '../../../hooks/useGetNodesNeedUpdate';
import { useAppSelector } from '../../../../../app/store/storeHooks';

export const WorkflowWarningTooltip = () => {
  const { t } = useTranslation();
  const nodesNeedUpdate = useGetNodesNeedUpdate();
  const isTouched = useAppSelector((s) => s.workflow.isTouched);

  return (
    <Flex flexDir="column" gap="2">
      {nodesNeedUpdate && (
        <Flex flexDir="column" gap="2">
          <Text fontWeight="semibold">{t('toast.loadedWithWarnings')}</Text>
          <Flex flexDir="column">
            <Text>{t('common.toResolve')}:</Text>
            <Text>
              {t('nodes.editMode')} &gt;&gt; {t('nodes.updateAllNodes')} &gt;&gt; {t('common.save')}
            </Text>
          </Flex>
        </Flex>
      )}

      {isTouched && (
        <Flex flexDir="column" gap="2">
          <Text fontWeight="semibold">{t('nodes.newWorkflowDesc2')}</Text>
          <Flex flexDir="column">
            <Text>{t('common.toResolve')}:</Text>
            <Text>
              {t('nodes.editMode')} &gt;&gt; {t('common.save')}
            </Text>
          </Flex>
        </Flex>
      )}
    </Flex>
  );
};
