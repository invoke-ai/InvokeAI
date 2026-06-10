import { Flex, Text } from '@chakra-ui/react';

import type { WidgetViewProps } from '../../types';

export const WorkflowWidgetView = ({ region }: WidgetViewProps) => {
  if (region === 'left') {
    return (
      <Text color="fg.subtle" fontSize="2xs">
        Workflow controls will render here when this widget is mounted into the left panel.
      </Text>
    );
  }

  return (
    <Flex align="center" h="full" justify="center" w="full">
      <Text color="fg.subtle" fontSize="sm">
        Workflow view
      </Text>
    </Flex>
  );
};
