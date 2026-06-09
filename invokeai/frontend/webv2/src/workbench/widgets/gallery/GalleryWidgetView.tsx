import { Flex, Stack, Text } from '@chakra-ui/react';
import { PiImageBold } from 'react-icons/pi';

import { StatusWidgetChip, WidgetPanelFrame } from '../../components/WidgetFrames';
import { WidgetFailureBoundary } from '../../components/WidgetFailureBoundary';
import type { WidgetViewProps } from '../../types';

export const GalleryWidgetView = ({ presentation, region }: WidgetViewProps) => {
  if (region === 'bottom') {
    if (presentation === 'expanded') {
      return (
        <Stack gap="2">
          <Text fontSize="xs" fontWeight="700">
            Gallery
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Gallery status and recent outputs will render here once gallery data is connected.
          </Text>
        </Stack>
      );
    }

    return <StatusWidgetChip icon={PiImageBold}>Gallery</StatusWidgetChip>;
  }

  if (region === 'right') {
    return (
      <WidgetFailureBoundary widgetId="gallery">
        <WidgetPanelFrame region="right">
          <Text fontSize="xs" fontWeight="700">
            Gallery
          </Text>
          <Text color="fg.subtle" fontSize="2xs">
            Gallery controls will render here when this widget is mounted into the right panel.
          </Text>
        </WidgetPanelFrame>
      </WidgetFailureBoundary>
    );
  }

  return (
    <WidgetFailureBoundary widgetId="gallery">
      <Flex align="center" bg="bg.canvas" h="full" justify="center" w="full">
        <Text color="fg.subtle" fontSize="sm">
          Gallery view
        </Text>
      </Flex>
    </WidgetFailureBoundary>
  );
};
