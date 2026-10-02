import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeApi,
  WidgetViewProps,
  WorkbenchRegion,
} from '@workbench/widgetContracts';

import { Box, Flex, Text } from '@chakra-ui/react';
import { getWidgetReadyMark, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui/Scrollable';
import { WidgetOverlayOwnerContext } from '@platform/ui/widgetOverlays';
import { WidgetSettingsButton } from '@workbench/settings/WidgetSettingsButton';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { memo, Suspense, use, useMemo } from 'react';

import { useWidgetRuntime } from './createWidgetRuntime';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';
import { WidgetHeader, WidgetHeaderActionsGroup, WidgetPanelFrame, WidgetTooltipFrame } from './WidgetFrames';
import { WidgetLoadingFallback } from './WidgetLoadingFallback';
import { areProjectWidgetRenderInstancesEqual } from './widgetRenderInstance';

interface WidgetRendererProps extends Omit<WidgetViewProps, 'manifest' | 'runtime'> {
  instance: WidgetInstanceContract;
  widget: RegisteredWidget;
}

interface WidgetRendererByIdProps extends Omit<WidgetViewProps, 'instance' | 'manifest' | 'runtime'> {
  instanceId: string;
  widget: RegisteredWidget;
}

/**
 * The chrome slots a region can hoist out of the widget frame. `actions` is the
 * full trailing cluster (widget actions, settings, float, overflow menu);
 * `viewActions` contains widget actions and settings, for chrome that already
 * carries its own layout controls.
 */
type WidgetChromeSlotName = 'actions' | 'label' | 'viewActions';

interface WidgetChromeSlotByIdProps {
  instanceId: string;
  /**
   * Specify the runtime region for hoisted chrome outside center so close/reveal commands target the correct
   * placement.
   */
  region?: WorkbenchRegion;
  slot: WidgetChromeSlotName;
  widget: RegisteredWidget;
}

export const WidgetRendererById = ({ instanceId, widget, ...props }: WidgetRendererByIdProps) => {
  const selection = useActiveProjectSelector(
    (project) => ({ instance: project.widgetInstances[instanceId], projectId: project.id }),
    areProjectWidgetRenderInstancesEqual
  );

  return selection.instance ? <WidgetRenderer instance={selection.instance} widget={widget} {...props} /> : null;
};

/**
 * Resolve widget chrome with the view's implementation/runtime for center and floating hosts. Callers supply
 * Suspense so other chrome paints during chunk loading.
 */
export const WidgetChromeSlotById = ({ instanceId, region = 'center', slot, widget }: WidgetChromeSlotByIdProps) => {
  const selection = useActiveProjectSelector(
    (project) => ({ instance: project.widgetInstances[instanceId], projectId: project.id }),
    areProjectWidgetRenderInstancesEqual
  );

  return selection.instance ? (
    <WidgetChromeSlot instance={selection.instance} region={region} slot={slot} widget={widget} />
  ) : null;
};

const WidgetChromeSlot = ({
  instance,
  region,
  slot,
  widget,
}: {
  instance: WidgetInstanceContract;
  region: WorkbenchRegion;
  slot: WidgetChromeSlotName;
  widget: RegisteredWidget;
}) => {
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const implementation = use(widget.implementation.load());
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
    getWidgetById,
    getWidgetsForRegion,
    instance: instanceMeta,
    project,
    region,
  });
  const HeaderActions = implementation.headerActions;
  const HeaderLabel = implementation.headerLabel;
  const actions = useMemo(
    () =>
      HeaderActions ? (
        <HeaderActions instance={instanceMeta} manifest={widget.manifest} region={region} runtime={runtime} />
      ) : null,
    [HeaderActions, instanceMeta, region, runtime, widget.manifest]
  );

  if (widget.manifest.chrome?.header === 'hidden') {
    return null;
  }

  // Custom labels replace only standard titles; renamed instances retain selector text.
  if (slot === 'label') {
    return HeaderLabel && !instanceMeta.title ? <HeaderLabel region={region} /> : null;
  }

  // Keep preferences beside floating widget actions without duplicating the window's layout controls.
  if (slot === 'viewActions') {
    return (
      <>
        {actions}
        <WidgetSettingsButton
          SettingsActions={implementation.settingsActions}
          instance={instanceMeta}
          manifest={widget.manifest}
          region={region}
          runtime={runtime}
        />
      </>
    );
  }

  return (
    <WidgetHeaderActionsGroup
      HeaderMenu={implementation.headerMenu}
      SettingsActions={implementation.settingsActions}
      actions={actions}
      instance={instanceMeta}
      manifest={widget.manifest}
      region={region}
      runtime={runtime}
    />
  );
};

export const WidgetRenderer = ({ instance, presentation, region, widget }: WidgetRendererProps) => {
  const projectId = useActiveProjectId();
  const loadingFallback = useMemo(
    () => <WidgetLoadingFallback instance={instance} presentation={presentation} region={region} widget={widget} />,
    [instance, presentation, region, widget]
  );
  const content = (
    <Suspense fallback={loadingFallback}>
      <LoadedWidgetRenderer instance={instance} presentation={presentation} region={region} widget={widget} />
    </Suspense>
  );

  if (!widget.manifest.failurePolicy.isolateRenderFailure) {
    return content;
  }

  return (
    <WidgetFailureBoundary
      instance={instance}
      presentation={presentation}
      projectId={projectId}
      region={region}
      resetKey={instance.id}
      widget={widget}
      widgetId={widget.manifest.id}
      onRetry={widget.implementation.retry}
    >
      {content}
    </WidgetFailureBoundary>
  );
};

const LoadedWidgetRenderer = ({ instance, presentation, region, widget }: WidgetRendererProps) => {
  const { getWidgetById, getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  const implementation = use(widget.implementation.load());
  const View = implementation.view;
  useMountEffect(() => {
    markSemanticReady(getWidgetReadyMark(region, instance.typeId));
  });
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
    getWidgetById,
    getWidgetsForRegion,
    instance: instanceMeta,
    project,
    region,
  });
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
    <WidgetOverlayOwnerContext value>
      <WidgetShellFrame
        implementation={implementation}
        instance={instanceMeta}
        presentation={presentation}
        region={region}
        runtime={runtime}
        widget={widget}
      >
        {content}
      </WidgetShellFrame>
    </WidgetOverlayOwnerContext>
  );
};

const WidgetShellFrame = ({
  children,
  implementation,
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  children: React.ReactNode;
  implementation: WidgetImplementation;
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) => {
  const safeContent = children;

  if (presentation === 'tooltip') {
    return <WidgetTooltipFrame icon={widget.manifest.icon}>{safeContent}</WidgetTooltipFrame>;
  }

  // Floating windows provide their own chrome (FloatingWidgetWindow), so the
  // view renders bare, like popovers and dialogs.
  if (region === 'popover' || region === 'dialog' || region === 'floating' || presentation === 'compact') {
    return safeContent;
  }

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame instanceId={instance.id} region={region} typeId={instance.typeId}>
        <HeaderSlot
          implementation={implementation}
          instance={instance}
          presentation={presentation}
          region={region}
          runtime={runtime}
          widget={widget}
        />
        <PanelBodySlot>{safeContent}</PanelBodySlot>
        <FooterSlot
          implementation={implementation}
          instance={instance}
          presentation={presentation}
          region={region}
          runtime={runtime}
          widget={widget}
        />
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
      <Box flex="1" minH="0" overflow="hidden" position="relative">
        {safeContent}
      </Box>
      <FooterSlot
        implementation={implementation}
        instance={instance}
        presentation={presentation}
        region={region}
        runtime={runtime}
        widget={widget}
      />
    </Flex>
  );
};

const areWidgetChromeInstancesEqual = (left: WidgetInstanceRuntimeMeta, right: WidgetInstanceRuntimeMeta): boolean =>
  left.id === right.id && left.typeId === right.typeId && left.title === right.title;

const areSlotPropsEqual = (
  left: {
    implementation: WidgetImplementation;
    instance: WidgetInstanceRuntimeMeta;
    presentation: WidgetViewProps['presentation'];
    region: WidgetViewProps['region'];
    runtime: WidgetRuntimeApi;
    widget: RegisteredWidget;
  },
  right: {
    implementation: WidgetImplementation;
    instance: WidgetInstanceRuntimeMeta;
    presentation: WidgetViewProps['presentation'];
    region: WidgetViewProps['region'];
    runtime: WidgetRuntimeApi;
    widget: RegisteredWidget;
  }
): boolean =>
  left.presentation === right.presentation &&
  left.implementation === right.implementation &&
  left.region === right.region &&
  left.widget === right.widget &&
  left.runtime === right.runtime &&
  areWidgetChromeInstancesEqual(left.instance, right.instance);

const HeaderSlot = memo(function HeaderSlot({
  implementation,
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  implementation: WidgetImplementation;
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) {
  const HeaderActions = implementation.headerActions;
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

  return (
    <Box bg="bg.subtle" flexShrink={0}>
      <WidgetHeader
        HeaderLabel={implementation.headerLabel}
        HeaderMenu={implementation.headerMenu}
        SettingsActions={implementation.settingsActions}
        actions={actions}
        instance={instance}
        manifest={widget.manifest}
        region={region}
        runtime={runtime}
      />
    </Box>
  );
}, areSlotPropsEqual);

// Stretch fill-height views while allowing flowing content to scroll vertically. minmax(0,1fr) prevents long
// content from widening and clipping panels.
const panelBodyContentProps = {
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 1fr)',
  maxW: 'full',
  minH: 'full',
} as const;

export const PanelBodySlot = ({ children }: { children: React.ReactNode }) => (
  <Scrollable contentProps={panelBodyContentProps} flex="1" minH="0" minW="0" overflowX="hidden">
    {children}
  </Scrollable>
);

const FooterSlot = memo(function FooterSlot({
  implementation,
  instance,
  presentation,
  region,
  runtime,
  widget,
}: {
  implementation: WidgetImplementation;
  instance: WidgetInstanceRuntimeMeta;
  presentation: WidgetViewProps['presentation'];
  region: WidgetViewProps['region'];
  runtime: WidgetRuntimeApi;
  widget: RegisteredWidget;
}) {
  const Footer = implementation.footer;

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

export const MissingWidgetFrame = ({ label, region }: { label: string; region: Exclude<WidgetRegion, 'center'> }) => (
  <WidgetPanelFrame region={region}>
    <Box p="3">
      <Text fontSize="xs" fontWeight="700">
        {label}
      </Text>
      <Text color="fg.subtle" fontSize="2xs">
        Widget view unavailable.
      </Text>
    </Box>
  </WidgetPanelFrame>
);
