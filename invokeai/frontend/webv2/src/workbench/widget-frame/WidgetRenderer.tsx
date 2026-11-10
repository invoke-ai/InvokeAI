import type { WidgetRegion } from '@workbench/layoutContracts';
import type {
  RegisteredWidget,
  WidgetImplementation,
  WidgetInstanceContract,
  WidgetInstanceId,
  WidgetInstanceRuntimeMeta,
  WidgetRuntimeApi,
  WidgetTypeId,
  WidgetViewProps,
  WorkbenchRegion,
} from '@workbench/widgetContracts';

import { Box, Flex, Text, VisuallyHidden } from '@chakra-ui/react';
import { getWidgetReadyMark, markSemanticReady } from '@platform/performance/semanticReady';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable } from '@platform/ui/Scrollable';
import { WidgetOverlayOwnerContext } from '@platform/ui/widgetOverlays';
import { WidgetSettingsButton } from '@workbench/settings/WidgetSettingsButton';
import { resolveWidgetInstanceLabel } from '@workbench/widgetLabels';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { memo, Suspense, use, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';

import { useWidgetRuntime } from './createWidgetRuntime';
import { WidgetFailureBoundary } from './WidgetFailureBoundary';
import { WidgetHeader, WidgetHeaderActionsGroup, WidgetPanelFrame, WidgetTooltipFrame } from './WidgetFrames';
import { WidgetIdentityIcon } from './WidgetIdentityIcon';
import { WidgetLoadingFallback } from './WidgetLoadingFallback';
import { useWidgetRecovery } from './widgetRecovery';
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
type WidgetChromeSlotName = 'actions' | 'label' | 'viewActions' | 'icon';

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

  const failureFallback = useMemo(
    () => (slot === 'icon' ? <WidgetIdentityIcon boxSize="3.5" icon={widget.manifest.icon} /> : undefined),
    [slot, widget.manifest.icon]
  );
  return selection.instance ? (
    <WidgetRecoveryBoundary
      instance={selection.instance}
      presentation="compact"
      region={region}
      widget={widget}
      fallback={failureFallback}
      slot={slot}
    >
      {slot === 'icon' ? (
        <WidgetImplementationIcon widget={widget} />
      ) : (
        <WidgetChromeSlot instance={selection.instance} region={region} slot={slot} widget={widget} />
      )}
    </WidgetRecoveryBoundary>
  ) : null;
};

const WidgetImplementationIcon = ({ widget }: { widget: RegisteredWidget }) => {
  use(widget.implementation.load());
  return <WidgetIdentityIcon boxSize="3.5" icon={widget.manifest.icon} />;
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

export const WidgetRenderer = ({ instance, presentation = 'expanded', region, widget }: WidgetRendererProps) => {
  const loadingFallback = useMemo(
    () => <WidgetLoadingFallback instance={instance} presentation={presentation} region={region} widget={widget} />,
    [instance, presentation, region, widget]
  );
  return (
    <WidgetRecoveryBoundary
      instance={instance}
      presentation={presentation}
      region={region}
      widget={widget}
      slot={presentation ?? 'expanded'}
    >
      <Suspense fallback={loadingFallback}>
        <LoadedWidgetRenderer instance={instance} presentation={presentation} region={region} widget={widget} />
      </Suspense>
    </WidgetRecoveryBoundary>
  );
};

const WidgetRecoveryBoundary = ({
  children,
  fallback,
  instance,
  presentation,
  region,
  widget,
  slot,
  framed,
}: Omit<WidgetRendererProps, 'instance'> & {
  instance: WidgetInstanceRuntimeMeta;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  slot: string;
  framed?: boolean;
}) => {
  const projectId = useActiveProjectId();
  const recovery = useWidgetRecovery(instance.id, widget, slot, region);
  const { t } = useTranslation();
  if (!widget.manifest.failurePolicy.isolateRenderFailure) {
    return children;
  }
  const boundary = (
    <WidgetFailureBoundary
      fallback={
        region === 'center' &&
        slot !== 'icon' &&
        slot !== presentation &&
        widget.implementation.getStatus() === 'failed'
          ? null
          : fallback
      }
      framed={framed}
      isRetrying={recovery.retrying || recovery.reloading}
      instance={instance}
      presentation={presentation}
      projectId={projectId}
      region={region}
      resetKey={`${projectId}:${instance.id}:${recovery.generation}`}
      widget={widget}
      widgetId={widget.manifest.id}
      onRetry={recovery.retry}
      onRequestReload={
        recovery.reloadAvailable && (slot === presentation || slot === 'body') ? recovery.requestReload : undefined
      }
      onFailure={recovery.fail}
      onRecovered={recovery.recover}
    >
      {children}
    </WidgetFailureBoundary>
  );
  const label = resolveWidgetInstanceLabel(instance, widget.manifest, t);
  return (
    <>
      {slot === presentation && presentation === 'expanded' ? (
        <WidgetRecoveryAnnouncement
          failed={recovery.hasFailure}
          label={label}
          generation={recovery.generation}
          retrying={recovery.retrying}
        />
      ) : null}
      {(region === 'popover' || region === 'dialog') && presentation === 'expanded' && slot === presentation ? (
        <Box
          role="group"
          aria-label={label}
          data-widget-recovery-host=""
          tabIndex={-1}
          display="flex"
          flex="1"
          flexDirection="column"
          focusVisibleRing="inside"
          minH="0"
          w="full"
        >
          {boundary}
        </Box>
      ) : (
        boundary
      )}
    </>
  );
};

export const WidgetRecoveryAnnouncement = ({
  failed,
  label,
  generation,
  retrying,
}: {
  failed: boolean;
  label: string;
  generation: number;
  retrying: boolean;
}) => {
  const { t } = useTranslation();
  const source = `${generation}:${retrying}:${failed}`;
  const [announced, setAnnounced] = useState({ source, text: '' });
  if (announced.source !== source) {
    setAnnounced({
      source,
      text: t(retrying ? 'widgets.failure.retrying' : failed ? 'widgets.failure.title' : 'widgets.failure.recovered', {
        label,
      }),
    });
  }
  return <VisuallyHidden role="status">{announced.text}</VisuallyHidden>;
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
        {presentation === 'compact' || presentation === 'tooltip' ? (
          content
        ) : (
          <WidgetRecoveryBoundary
            instance={instance}
            presentation={presentation}
            region={region}
            widget={widget}
            slot="body"
            framed={false}
          >
            {content}
          </WidgetRecoveryBoundary>
        )}
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
  const { t } = useTranslation();

  if (presentation === 'tooltip') {
    return <WidgetTooltipFrame icon={widget.manifest.icon}>{children}</WidgetTooltipFrame>;
  }

  // Floating windows provide their own chrome (FloatingWidgetWindow), so the
  // view renders bare, like popovers and dialogs.
  if (region === 'popover' || region === 'dialog' || region === 'floating' || presentation === 'compact') {
    return children;
  }

  const footer = implementation.footer ? (
    <WidgetRecoveryBoundary instance={instance} presentation="compact" region={region} widget={widget} slot="footer">
      <FooterSlot
        implementation={implementation}
        instance={instance}
        presentation={presentation}
        region={region}
        runtime={runtime}
        widget={widget}
      />
    </WidgetRecoveryBoundary>
  ) : null;

  if (region === 'left' || region === 'right' || region === 'bottom') {
    return (
      <WidgetPanelFrame
        instanceId={instance.id}
        label={resolveWidgetInstanceLabel(instance, widget.manifest, t)}
        region={region}
        typeId={instance.typeId}
      >
        {widget.manifest.chrome?.header === 'hidden' ? null : (
          <WidgetRecoveryBoundary
            instance={instance}
            presentation="compact"
            region={region}
            widget={widget}
            slot="header"
          >
            <HeaderSlot
              implementation={implementation}
              instance={instance}
              presentation={presentation}
              region={region}
              runtime={runtime}
              widget={widget}
            />
          </WidgetRecoveryBoundary>
        )}
        <PanelBodySlot>{children}</PanelBodySlot>
        {footer}
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
        {children}
      </Box>
      {footer}
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

/**
 * Stands in for a widget whose view cannot render. It carries the instance's identity like any other frame, so focus
 * moves and hotkeys that look for the instance find it here.
 */
export const MissingWidgetFrame = ({
  instanceId,
  label,
  region,
  typeId,
}: {
  instanceId: WidgetInstanceId;
  label: string;
  region: Exclude<WidgetRegion, 'center'>;
  /** Unknown when the instance itself is gone. */
  typeId: WidgetTypeId | undefined;
}) => {
  const { t } = useTranslation();

  return (
    <WidgetPanelFrame instanceId={instanceId} label={label} region={region} typeId={typeId}>
      <Box p="3">
        <Text fontSize="md" fontWeight="700">
          {label}
        </Text>
        <Text color="fg.subtle" fontSize="xs">
          {t('common.widgetViewUnavailable')}
        </Text>
      </Box>
    </WidgetPanelFrame>
  );
};
