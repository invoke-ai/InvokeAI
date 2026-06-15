import { Box, Flex, Text } from '@chakra-ui/react';

import type { RegisteredWidget, WidgetViewProps } from '@workbench/types';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';
import { WidgetHeader, WidgetPanelFrame } from './WidgetFrames';

interface WidgetRendererProps extends Omit<WidgetViewProps, 'manifest'> {
  widget: RegisteredWidget;
}

export const WidgetRenderer = ({ presentation, region, widget }: WidgetRendererProps) => {
  const View = widget.manifest.view;

  if (!View) {
    return null;
  }

  const content = <View manifest={widget.manifest} presentation={presentation} region={region} />;
  const framedContent = (
    <WidgetShellFrame content={content} presentation={presentation} region={region} widget={widget} />
  );

  if (!widget.manifest.failurePolicy.isolateRenderFailure) {
    return framedContent;
  }

  return <WidgetFailureBoundary widgetId={widget.manifest.id}>{framedContent}</WidgetFailureBoundary>;
};

const WidgetShellFrame = ({
  content,
  presentation,
  region,
  widget,
}: {
  content: React.ReactNode;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  widget: RegisteredWidget;
}) => {
  if (region === 'popover' || region === 'dialog' || presentation === 'compact' || presentation === 'tooltip') {
    return content;
  }

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame region={region}>
        <HeaderSlot presentation={presentation} region={region} widget={widget} />
        <PanelBodySlot>{content}</PanelBodySlot>
        <FooterSlot presentation={presentation} region={region} widget={widget} />
      </WidgetPanelFrame>
    );
  }

  return (
    <Flex bg="bg.inset" direction="column" h="full" minH="0" w="full">
      <HeaderSlot presentation={presentation} region={region} widget={widget} />
      <Box flex="1" minH="0" overflow="hidden" position="relative">
        {content}
      </Box>
      <FooterSlot presentation={presentation} region={region} widget={widget} />
    </Flex>
  );
};

const HeaderSlot = ({
  presentation,
  region,
  widget,
}: {
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  widget: RegisteredWidget;
}) => {
  if (widget.manifest.chrome?.header === 'hidden') {
    return null;
  }

  const HeaderActions = widget.manifest.headerActions;
  const actions = HeaderActions ? (
    <HeaderActions manifest={widget.manifest} presentation={presentation} region={region} />
  ) : null;
  const bg = region === 'center' ? 'bg' : 'bg.subtle';

  return (
    <Box bg={bg} flexShrink={0}>
      <WidgetHeader actions={actions} manifest={widget.manifest} region={region} />
    </Box>
  );
};

const PanelBodySlot = ({ children }: { children: React.ReactNode }) => (
  <Flex direction="column" flex="1" minH="0" minW="0" overflowX="hidden" overflowY="auto">
    {children}
  </Flex>
);

const FooterSlot = ({
  presentation,
  region,
  widget,
}: {
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  widget: RegisteredWidget;
}) => {
  const Footer = widget.manifest.footer;

  if (!Footer) {
    return null;
  }
  const bg = region === 'center' ? 'bg.inset' : 'bg.subtle';

  return (
    <Box bg={bg} flexShrink={0}>
      <Footer manifest={widget.manifest} presentation={presentation} region={region} />
    </Box>
  );
};

export const MissingWidgetFrame = ({ label, region }: { label: string; region: 'bottom' | 'left' | 'right' }) => (
  <WidgetPanelFrame region={region}>
    <Text fontSize="xs" fontWeight="700">
      {label}
    </Text>
    <Text color="fg.subtle" fontSize="2xs">
      Widget view unavailable.
    </Text>
  </WidgetPanelFrame>
);
