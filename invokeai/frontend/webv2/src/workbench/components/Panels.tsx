import type { WidgetId, WorkbenchRegion } from '@workbench/types';
import { getWidgetById } from '@workbench/widgetRegistry';
import { MissingWidgetFrame, WidgetRenderer } from './WidgetRenderer';

/** Left panel — hosts the active registered widget panel view. */
export const LeftPanel = ({ widgetId }: { widgetId: WidgetId }) => (
  <WidgetPanelSlot widgetId={widgetId} panel="leftPanel" />
);

/** Right panel — hosts the active registered widget panel view. */
export const RightPanel = ({ widgetId }: { widgetId: WidgetId }) => (
  <WidgetPanelSlot widgetId={widgetId} panel="rightPanel" />
);

const panelRegions = {
  leftPanel: 'left',
  rightPanel: 'right',
} as const satisfies Record<string, WorkbenchRegion>;

const WidgetPanelSlot = ({ widgetId, panel }: { widgetId: WidgetId; panel: keyof typeof panelRegions }) => {
  const widget = getWidgetById(widgetId);
  const View = widget?.manifest.view;
  const region = panelRegions[panel];

  if (!widget || widget.status !== 'enabled' || !View) {
    return <MissingWidgetFrame label={widget?.manifest.labelText ?? widgetId} region={region} />;
  }

  return <WidgetRenderer widget={widget} region={region} />;
};
