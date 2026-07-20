export const loadWorkflowWidgetImplementation = () =>
  import('./ui/implementation').then((module) => module.widgetImplementation);
