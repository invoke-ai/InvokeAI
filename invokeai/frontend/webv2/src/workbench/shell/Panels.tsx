import type { WidgetInstanceId, WorkbenchRegion } from '@workbench/widgetContracts';

import { MissingWidgetFrame, WidgetRendererById } from '@workbench/widget-frame';
import { areWidgetRenderInstancesEqual } from '@workbench/widget-frame/widgetRenderInstance';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useActiveProjectId, useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { Activity } from 'react';
import { useTranslation } from 'react-i18next';

import {
  areInstanceIdListsEqual,
  getActiveInstanceIdsOutside,
  useMountedInstanceIds,
  withoutInstancesShownElsewhere,
} from './useMountedInstanceIds';

/** Left panel — hosts the active registered widget panel view. */
export const LeftPanel = ({ instanceId }: { instanceId: WidgetInstanceId }) => (
  <WidgetPanelSlot instanceId={instanceId} panel="leftPanel" />
);

/** Right panel — hosts the active registered widget panel view. */
export const RightPanel = ({ instanceId }: { instanceId: WidgetInstanceId }) => (
  <WidgetPanelSlot instanceId={instanceId} panel="rightPanel" />
);

const panelRegions = {
  leftPanel: 'left',
  rightPanel: 'right',
} as const satisfies Record<string, WorkbenchRegion>;

/**
 * Keeps the panel widgets this session has already shown mounted behind the
 * active one, so switching a layout preset hides them rather than destroying
 * their scroll position, selection and virtualizer state. The remembered set is
 * independent of the region's `instanceIds`, which a preset replaces wholesale.
 */
const WidgetPanelSlot = ({ instanceId, panel }: { instanceId: WidgetInstanceId; panel: keyof typeof panelRegions }) => {
  const projectId = useActiveProjectId();
  const activeIdsElsewhere = useActiveProjectSelector(
    (project) => getActiveInstanceIdsOutside(project.widgetRegions, panelRegions[panel]),
    areInstanceIdListsEqual
  );
  const mountedIds = withoutInstancesShownElsewhere(
    useMountedInstanceIds(instanceId, projectId),
    instanceId,
    activeIdsElsewhere
  );

  return (
    <>
      {mountedIds.map((id) => (
        <Activity key={id} mode={id === instanceId ? 'visible' : 'hidden'}>
          <WidgetPanelInstance instanceId={id} panel={panel} />
        </Activity>
      ))}
    </>
  );
};

const WidgetPanelInstance = ({
  instanceId,
  panel,
}: {
  instanceId: WidgetInstanceId;
  panel: keyof typeof panelRegions;
}) => {
  const { t } = useTranslation();
  const instance = useActiveProjectSelector(
    (project) => project.widgetInstances[instanceId],
    areWidgetRenderInstancesEqual
  );
  const widget = instance ? getWidgetById(instance.typeId) : undefined;
  const region = panelRegions[panel];

  if (!instance || !widget || widget.status !== 'enabled') {
    return <MissingWidgetFrame label={widget ? resolveWidgetLabel(widget.manifest, t) : instanceId} region={region} />;
  }

  return <WidgetRendererById instanceId={instance.id} widget={widget} region={region} />;
};
