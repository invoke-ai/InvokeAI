import type { RegisteredWidget, WidgetInstanceContract, WidgetRuntimeApi, WidgetViewProps } from '@workbench/types';

import { Box, Flex, Text } from '@chakra-ui/react';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';

import { useWidgetRuntime } from './createWidgetRuntime';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';
import { WidgetHeader, WidgetPanelFrame } from './WidgetFrames';

interface WidgetRendererProps extends Omit<WidgetViewProps, 'manifest' | 'runtime'> {
  instance: WidgetInstanceContract;
  widget: RegisteredWidget;
}

export const WidgetRenderer = ({ instance, presentation, region, widget }: WidgetRendererProps) => {
  const dispatch = useWorkbenchDispatch();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const View = widget.manifest.view;
  const runtime = useWidgetRuntime({ dispatch, getWidgetById, getWidgetsForRegion, instance, project, region });

  if (!View) {
    return null;
  }

  const content = (
    <View
      instance={instance}
      manifest={widget.manifest}
      presentation={presentation}
      region={region}
      runtime={runtime}
    />
  );

  return (
    <WidgetShellFrame
      content={content}
      instance={instance}
      presentation={presentation}
      region={region}
      runtime={runtime}
      widget={widget}
    />
  );
};

const WidgetShellFrame = ({
  content,
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  content: React.ReactNode;
  instance: WidgetInstanceContract;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) => {
  const safeContent = (
    <RenderFailureSlot instance={instance} widget={widget}>
      {content}
    </RenderFailureSlot>
  );

  if (region === 'popover' || region === 'dialog' || presentation === 'compact' || presentation === 'tooltip') {
    return safeContent;
  }

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame region={region}>
        <HeaderSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
        <PanelBodySlot>{safeContent}</PanelBodySlot>
        <FooterSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
      </WidgetPanelFrame>
    );
  }

  return (
    <Flex bg="bg.inset" direction="column" h="full" minH="0" w="full">
      <HeaderSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
      <Box flex="1" minH="0" overflow="hidden" position="relative">
        {safeContent}
      </Box>
      <FooterSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
    </Flex>
  );
};

const RenderFailureSlot = ({
  children,
  instance,
  widget,
}: {
  children: React.ReactNode;
  instance: WidgetInstanceContract;
  widget: RegisteredWidget;
}) => {
  if (!widget.manifest.failurePolicy.isolateRenderFailure) {
    return children;
  }

  return (
    <WidgetFailureBoundary resetKey={instance.id} widgetId={widget.manifest.id}>
      {children}
    </WidgetFailureBoundary>
  );
};

const HeaderSlot = ({
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  instance: WidgetInstanceContract;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) => {
  if (widget.manifest.chrome?.header === 'hidden') {
    return null;
  }

  const HeaderActions = widget.manifest.headerActions;
  const actions = HeaderActions ? (
    <HeaderActions
      instance={instance}
      manifest={widget.manifest}
      presentation={presentation}
      region={region}
      runtime={runtime}
    />
  ) : null;
  const bg = region === 'center' ? 'bg' : 'bg.subtle';

  return (
    <Box bg={bg} flexShrink={0}>
      <WidgetHeader
        actions={actions}
        instance={instance}
        manifest={widget.manifest}
        region={region}
        runtime={runtime}
      />
    </Box>
  );
};

const PanelBodySlot = ({ children }: { children: React.ReactNode }) => (
  <Flex direction="column" flex="1" minH="0" minW="0" overflowX="hidden" overflowY="auto">
    {children}
  </Flex>
);

const FooterSlot = ({
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  instance: WidgetInstanceContract;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) => {
  const Footer = widget.manifest.footer;

  if (!Footer) {
    return null;
  }
  const bg = region === 'center' ? 'bg.inset' : 'bg.subtle';

  return (
    <Box bg={bg} flexShrink={0}>
      <Footer
        instance={instance}
        manifest={widget.manifest}
        presentation={presentation}
        region={region}
        runtime={runtime}
      />
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
