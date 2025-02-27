import { Flex, Text } from '@invoke-ai/ui-library';
import dateFormat, { masks } from 'dateformat';
import { useTranslation } from 'react-i18next';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

export const WorkflowListItemTooltip = ({ workflow }: { workflow: WorkflowRecordListItemDTO }) => {
  const { t } = useTranslation();
  return (
    <Flex flexDir="column" gap={1}>
      <Text>{workflow.description}</Text>
      {workflow.category !== 'default' && (
        <Flex flexDir="column">
          <Text>
            {t('workflows.opened')}: {dateFormat(workflow.opened_at, masks.shortDate)}
          </Text>
          <Text>
            {t('common.updated')}: {dateFormat(workflow.updated_at, masks.shortDate)}
          </Text>
          <Text>
            {t('common.created')}: {dateFormat(workflow.created_at, masks.shortDate)}
          </Text>
        </Flex>
      )}
    </Flex>
  );
};
