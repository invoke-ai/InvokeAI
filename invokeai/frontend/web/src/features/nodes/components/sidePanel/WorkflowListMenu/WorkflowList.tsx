import { Button, Collapse, Flex, Icon, Text } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IAINoContentFallback } from 'common/components/IAIImageFallback';
import { useCategorySections } from 'features/nodes/hooks/useCategorySections';
import { selectWorkflowSearchTerm } from 'features/nodes/store/workflowSlice';
import { useTranslation } from 'react-i18next';
import { PiCaretDownBold } from 'react-icons/pi';
import type { WorkflowRecordListItemDTO } from 'services/api/types';

import { WorkflowListItem } from './WorkflowListItem';

export const WorkflowList = ({ title, data }: { title: string; data: WorkflowRecordListItemDTO[] }) => {
  const { t } = useTranslation();
  const searchTerm = useAppSelector(selectWorkflowSearchTerm);

  const { isOpen, onToggle } = useCategorySections(title);

  return (
    <Flex flexDir="column">
      <Button variant="unstyled" onClick={onToggle}>
        <Flex gap={2} alignItems="center">
          <Icon boxSize={4} as={PiCaretDownBold} transform={isOpen ? undefined : 'rotate(-90deg)'} fill="base.500" />
          <Text fontSize="sm" fontWeight="semibold" userSelect="none" color="base.500">
            {title}
          </Text>
        </Flex>
      </Button>
      <Collapse in={isOpen}>
        {data.length ? (
          data.map((workflow) => <WorkflowListItem workflow={workflow} key={workflow.workflow_id} />)
        ) : (
          <IAINoContentFallback
            fontSize="sm"
            py={4}
            label={searchTerm ? t('stylePresets.noMatchingTemplates') : t('stylePresets.noTemplates')}
            icon={null}
          />
        )}
      </Collapse>
    </Flex>
  );
};
