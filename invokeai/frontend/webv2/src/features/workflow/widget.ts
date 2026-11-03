export const loadWorkflowWidgetImplementation = () =>
  import('./ui/implementation').then((module) => module.widgetImplementation);

export const loadWorkflowWidgetHost = () =>
  import('./ui/WorkflowWidgetChrome').then((module) => module.WorkflowDialogHost);
