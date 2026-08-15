import type {
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeApi,
  WidgetViewProps,
} from '@workbench/widgetContracts';

import { Box, Flex, Text } from '@chakra-ui/react';
import { getWidgetReadyMark, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
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

/** The chrome slots the center region hoists out of the widget frame. */
type WidgetChromeSlotName = 'actions' | 'label';

interface WidgetChromeSlotByIdProps {
  instanceId: string;
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
 * Renders one chrome slot of a widget outside its own frame. The center region
 * floats its chrome over the work surface, so the label and action slots are
 * mounted by `CenterArea` rather than by the widget's frame — this resolves the
 * same implementation and runtime the view would get.
 *
 * Suspends on first load of the implementation chunk; callers wrap it in their
 * own `Suspense` so the rest of the chrome paints immediately.
 */
export const WidgetChromeSlotById = ({ instanceId, slot, widget }: WidgetChromeSlotByIdProps) => {
  const selection = useActiveProjectSelector(
    (project) => ({ instance: project.widgetInstances[instanceId], projectId: project.id }),
    areProjectWidgetRenderInstancesEqual
  );

  return selection.instance ? <WidgetChromeSlot instance={selection.instance} slot={slot} widget={widget} /> : null;
};

const WidgetChromeSlot = ({
  instance,
  slot,
  widget,
}: {
  instance: WidgetInstanceContract;
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
    region: 'center',
  });
  const HeaderActions = implementation.headerActions;
  const HeaderLabel = implementation.headerLabel;
  const actions = useMemo(
    () =>
      HeaderActions ? (
        <HeaderActions instance={instanceMeta} manifest={widget.manifest} region="center" runtime={runtime} />
      ) : null,
    [HeaderActions, instanceMeta, runtime, widget.manifest]
  );

  if (widget.manifest.chrome?.header === 'hidden') {
    return null;
  }

  // A widget-supplied label only replaces the standard title, which the center
  // view selector already shows — a renamed instance keeps the selector's text.
  if (slot === 'label') {
    return HeaderLabel && !instanceMeta.title ? <HeaderLabel region="center" /> : null;
  }

  return (
    <WidgetHeaderActionsGroup
      HeaderMenu={implementation.headerMenu}
      actions={actions}
      instance={instanceMeta}
      manifest={widget.manifest}
      region="center"
      runtime={runtime}
    />
  );
};

export const WidgetRenderer = ({ instance, presentation, region, widget }: WidgetRendererProps) => {
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

  // The center region floats its chrome over the work surface, so the widget's
  // header slots are mounted by `CenterArea` and the body runs full-bleed here.
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
        actions={actions}
        instance={instance}
        manifest={widget.manifest}
        region={region}
        runtime={runtime}
      />
    </Box>
  );
}, areSlotPropsEqual);

// Grid content with minH="full" stretches fill-height widget views (gallery,
// layers, preview) to the viewport while letting flowing views grow and scroll.
// The explicit minmax(0, 1fr) column is load-bearing: an implicit auto track
// sizes to its item's max-content, so any long unbreakable text or wide row
// inside a widget (prompt strings, UUIDs, button groups) silently stretches
// the widget past its panel and clips. Panels only ever scroll vertically.
// Covered by PanelBodySlot.browser.test.tsx.
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
