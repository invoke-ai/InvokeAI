import type { WidgetInstanceId, WorkbenchRegion } from '@workbench/widgetContracts';

import { MissingWidgetFrame, WidgetRendererById } from '@workbench/widget-frame';
import { areWidgetRenderInstancesEqual } from '@workbench/widget-frame/widgetRenderInstance';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { useTranslation } from 'react-i18next';

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

const WidgetPanelSlot = ({ instanceId, panel }: { instanceId: WidgetInstanceId; panel: keyof typeof panelRegions }) => {
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
