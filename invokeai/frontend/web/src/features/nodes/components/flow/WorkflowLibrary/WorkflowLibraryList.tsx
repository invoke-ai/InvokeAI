import { Flex, Spacer, Text } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaFolderOpen, FaTrash } from 'react-icons/fa';
import { paths } from 'services/api/schema';
import ScrollableContent from 'features/nodes/components/sidePanel/ScrollableContent';

type Props = {
  data: paths['/api/v1/workflows/']['get']['responses']['200']['content']['application/json'];
};

const WorkflowLibraryList = ({ data }: Props) => {
  const { t } = useTranslation();

  return (
    <Flex w="full" h="full" layerStyle="second" p={2} borderRadius="base">
      <ScrollableContent>
        <Flex w="full" h="full" gap={2} flexDir="column">
          {data.items.map((w) => (
            <Flex key={w.workflow_id} w="full">
              <Flex w="full" alignItems="center" gap={2}>
                <Text>{w.workflow_id}</Text>
                <Spacer />
                <IAIIconButton
                  icon={<FaFolderOpen />}
                  aria-label={t('workflows.loadWorkflow')}
                  tooltip={t('workflows.loadWorkflow')}
                />
                <IAIIconButton
                  icon={<FaTrash />}
                  colorScheme="error"
                  aria-label={t('workflows.deleteWorkflow')}
                  tooltip={t('workflows.deleteWorkflow')}
                />
              </Flex>
            </Flex>
          ))}
        </Flex>
      </ScrollableContent>
    </Flex>
  );
};

export default memo(WorkflowLibraryList);
