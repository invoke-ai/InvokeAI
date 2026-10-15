/** Queue-owned deferred widget slots; Workbench supplies the manifest adapter. */
export const loadQueueWidgetImplementation = () =>
  import('./ui').then((module) => ({
    footer: module.ModelCacheFooter,
    headerActions: module.QueueHeaderActions,
    headerLabel: module.QueueHeaderLabel,
    headerMenu: module.QueueHeaderMenu,
    view: module.QueueWidgetView,
  }));

export const loadQueueWidgetHost = () => import('./ui/QueueDataRuntime').then((module) => module.QueueDataRuntime);
