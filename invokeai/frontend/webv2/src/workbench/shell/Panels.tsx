import type { WidgetInstanceId, WorkbenchRegion } from '@workbench/types';

import { MissingWidgetFrame, WidgetRenderer } from '@workbench/widget-frame';
import { getWidgetById } from '@workbench/widgetRegistry';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';

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
  const instance = useActiveProjectSelector((project) => project.widgetInstances[instanceId]);
  const widget = instance ? getWidgetById(instance.typeId) : undefined;
  const View = widget?.manifest.view;
  const region = panelRegions[panel];

  if (!instance || !widget || widget.status !== 'enabled' || !View) {
    return <MissingWidgetFrame label={widget?.manifest.labelText ?? instanceId} region={region} />;
  }

  return <WidgetRenderer instance={instance} widget={widget} region={region} />;
};
