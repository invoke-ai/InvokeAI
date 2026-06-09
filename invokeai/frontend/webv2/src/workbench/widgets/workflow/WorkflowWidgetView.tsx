import { Flex, Text } from '@chakra-ui/react';

import { GraphBearingWidgetHeader, WidgetPanelFrame } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import type { WidgetViewProps } from '../../types';

export const WorkflowWidgetView = ({ manifest, region }: WidgetViewProps) => {
  if (region === 'left') {
    return (
      <WidgetFailureBoundary widgetId="workflow">
        <WidgetPanelFrame region="left">
          <GraphBearingWidgetHeader manifest={manifest} region="left" />
          <Text color="fg.subtle" fontSize="2xs">
            Workflow controls will render here when this widget is mounted into the left panel.
          </Text>
        </WidgetPanelFrame>
      </WidgetFailureBoundary>
    );
  }

  return (
    <WidgetFailureBoundary widgetId="workflow">
      <Flex align="center" bg="bg.canvas" h="full" justify="center" position="relative" w="full">
        <Flex position="absolute" right="2" top="2" zIndex="2">
          <GraphBearingWidgetHeader manifest={manifest} region="center" />
        </Flex>
        <Text color="fg.subtle" fontSize="sm">
          Workflow view
        </Text>
      </Flex>
    </WidgetFailureBoundary>
  );
};
