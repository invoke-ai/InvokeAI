import {
  WorkflowDialogHost,
  WorkflowHeaderActions,
  WorkflowMenuItems,
  WorkflowWidgetLabel,
} from './WorkflowWidgetChrome';
import { WorkflowWidgetView } from './WorkflowWidgetView';

export const widgetImplementation = {
  headerActions: WorkflowHeaderActions,
  headerLabel: WorkflowWidgetLabel,
  headerMenu: WorkflowMenuItems,
  host: WorkflowDialogHost,
  view: WorkflowWidgetView,
};
