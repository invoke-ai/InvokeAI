import type {
  RegisteredWidget,
  WidgetInstanceContract,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeApi,
  WidgetViewProps,
} from '@workbench/types';

import { Box, Flex, Text } from '@chakra-ui/react';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector, useWorkbenchDispatch } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { memo, useMemo } from 'react';

import { useWidgetRuntime } from './createWidgetRuntime';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';
import { WidgetHeader, WidgetPanelFrame } from './WidgetFrames';
import { areProjectWidgetRenderInstancesEqual } from './widgetRenderInstance';

interface WidgetRendererProps extends Omit<WidgetViewProps, 'manifest' | 'runtime'> {
  instance: WidgetInstanceContract;
  widget: RegisteredWidget;
}

interface WidgetRendererByIdProps extends Omit<WidgetViewProps, 'instance' | 'manifest' | 'runtime'> {
  instanceId: string;
  widget: RegisteredWidget;
}

export const WidgetRendererById = ({ instanceId, widget, ...props }: WidgetRendererByIdProps) => {
  const selection = useActiveProjectSelector(
    (project) => ({ instance: project.widgetInstances[instanceId], projectId: project.id }),
    areProjectWidgetRenderInstancesEqual
  );

  return selection.instance ? <WidgetRenderer instance={selection.instance} widget={widget} {...props} /> : null;
};

export const WidgetRenderer = ({ instance, presentation, region, widget }: WidgetRendererProps) => {
  const dispatch = useWorkbenchDispatch();
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const View = widget.manifest.view;
  const instanceMeta: WidgetInstanceRuntimeMeta = useMemo(
    () => ({
      createdAt: instance.createdAt,
      id: instance.id,
      title: instance.title,
      typeId: instance.typeId,
    }),
    [instance.createdAt, instance.id, instance.title, instance.typeId]
  );
  const runtime = useWidgetRuntime({
    dispatch,
    getWidgetById,
    getWidgetsForRegion,
    instance: instanceMeta,
    project,
    region,
  });

  if (!View) {
    return null;
  }

  const content = (
    <View
      instance={instanceMeta}
      manifest={widget.manifest}
      presentation={presentation}
      region={region}
      runtime={runtime}
    />
  );

  return (
    <WidgetShellFrame
      instance={instanceMeta}
      presentation={presentation}
      region={region}
      runtime={runtime}
      widget={widget}
    >
      {content}
    </WidgetShellFrame>
  );
};

const WidgetShellFrame = ({
  children,
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  children: React.ReactNode;
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) => {
  const safeContent = (
    <RenderFailureSlot instance={instance} widget={widget}>
      {children}
    </RenderFailureSlot>
  );

  if (region === 'popover' || region === 'dialog' || presentation === 'compact' || presentation === 'tooltip') {
    return safeContent;
  }

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame instanceId={instance.id} region={region} typeId={instance.typeId}>
        <HeaderSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
        <PanelBodySlot>{safeContent}</PanelBodySlot>
        <FooterSlot instance={instance} presentation={presentation} region={region} runtime={runtime} widget={widget} />
      </WidgetPanelFrame>
    );
  }

  return (
    <Flex
      bg="bg.inset"
      data-hotkey-widget-instance-id={instance.id}
      data-hotkey-widget-region={region}
      data-hotkey-widget-type-id={instance.typeId}
      direction="column"
      h="full"
      minH="0"
      w="full"
    >
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
  instance: WidgetInstanceRuntimeMeta;
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

const areWidgetChromeInstancesEqual = (left: WidgetInstanceRuntimeMeta, right: WidgetInstanceRuntimeMeta): boolean =>
  left.id === right.id && left.typeId === right.typeId && left.title === right.title;

const areSlotPropsEqual = (
  left: {
    instance: WidgetInstanceRuntimeMeta;
    presentation: WidgetViewProps['presentation'];
    region: WidgetViewProps['region'];
    runtime: WidgetRuntimeApi;
    widget: RegisteredWidget;
  },
  right: {
    instance: WidgetInstanceRuntimeMeta;
    presentation: WidgetViewProps['presentation'];
    region: WidgetViewProps['region'];
    runtime: WidgetRuntimeApi;
    widget: RegisteredWidget;
  }
): boolean =>
  left.presentation === right.presentation &&
  left.region === right.region &&
  left.widget === right.widget &&
  left.runtime === right.runtime &&
  areWidgetChromeInstancesEqual(left.instance, right.instance);

const HeaderSlot = memo(function HeaderSlot({
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) {
  const HeaderActions = widget.manifest.headerActions;
  const actions = useMemo(
    () =>
      HeaderActions ? (
        <HeaderActions
          instance={instance}
          manifest={widget.manifest}
          presentation={presentation}
          region={region}
          runtime={runtime}
        />
      ) : null,
    [HeaderActions, instance, presentation, region, runtime, widget.manifest]
  );

  if (widget.manifest.chrome?.header === 'hidden') {
    return null;
  }
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
}, areSlotPropsEqual);

const PanelBodySlot = ({ children }: { children: React.ReactNode }) => (
  <Flex direction="column" flex="1" minH="0" minW="0" overflowX="hidden" overflowY="auto">
    {children}
  </Flex>
);

const FooterSlot = memo(function FooterSlot({
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) {
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
}, areSlotPropsEqual);

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
