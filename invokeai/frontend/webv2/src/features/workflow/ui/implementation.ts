import { WorkflowHeaderActions, WorkflowMenuItems, WorkflowWidgetLabel } from './WorkflowWidgetChrome';
import { WorkflowWidgetView } from './WorkflowWidgetView';

export const widgetImplementation = {
  headerActions: WorkflowHeaderActions,
  headerLabel: WorkflowWidgetLabel,
  headerMenu: WorkflowMenuItems,
  view: WorkflowWidgetView,
};
