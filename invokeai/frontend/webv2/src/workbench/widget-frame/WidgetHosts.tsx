import { getWidgetHosts } from '@workbench/widgetRegistry';
import { useWorkbenchSelector } from '@workbench/WorkbenchContext';
import { Suspense, use } from 'react';

import { WidgetFailureBoundary } from './WidgetFailureBoundary';

/** What a project shows: its resolved layout. */
type WidgetHostLayout = {
  floatingWidgets?: Record<string, unknown>;
  widgetInstances: Record<string, { typeId?: string }>;
  widgetRegions: Record<string, { instanceIds: string[] }>;
};

type WidgetHostWorkflows = { entries: Array<{ document: { nodes: unknown[] }; source?: unknown }> };

export const projectHasWidgetType = (layout: WidgetHostLayout, widgetTypeId: string): boolean => {
  return (
    Object.values(layout.widgetRegions).some((region) =>
      region.instanceIds.some((instanceId) => layout.widgetInstances[instanceId]?.typeId === widgetTypeId)
    ) ||
    Object.keys(layout.floatingWidgets ?? {}).some(
      (instanceId) => layout.widgetInstances[instanceId]?.typeId === widgetTypeId
    )
  );
};

// Documents are immutable per identity; only an edited workflow is rescanned.
const savedWorkflowCallScans = new WeakMap<object, boolean>();

const documentCallsSavedWorkflow = (document: { nodes: readonly unknown[] }): boolean => {
  let calls = savedWorkflowCallScans.get(document);

  if (calls === undefined) {
    calls = document.nodes.some((node) => {
      if (typeof node !== 'object' || node === null) {
        return false;
      }

      const candidate = node as { type?: unknown; data?: { type?: unknown } };
      return candidate.type === 'invocation' && candidate.data?.type === 'call_saved_workflow';
    });
    savedWorkflowCallScans.set(document, calls);
  }

  return calls;
};

/** Saved-workflow calls reconcile against live library records even while the editor is closed. */
export const projectNeedsWorkflowHost = (layout: WidgetHostLayout, workflows?: WidgetHostWorkflows): boolean =>
  projectHasWidgetType(layout, 'workflow') ||
  (workflows?.entries.some((entry) => documentCallsSavedWorkflow(entry.document)) ?? false);

const WidgetHost = ({ widget }: { widget: ReturnType<typeof getWidgetHosts>[number] }) => {
  const Host = use(widget.host!.load());

  // Host is the deferred resource's stable cached export; React Compiler misidentifies a direct use() result as a new
  // component.
  // eslint-disable-next-line react/static-components
  return <Host />;
};

const WidgetHostBoundary = ({ widget }: { widget: ReturnType<typeof getWidgetHosts>[number] }) => {
  const content = (
    <Suspense fallback={null}>
      <WidgetHost widget={widget} />
    </Suspense>
  );

  return widget.manifest.failurePolicy.isolateRenderFailure ? (
    <WidgetFailureBoundary
      resetKey={widget.manifest.id}
      widget={widget}
      widgetId={widget.manifest.id}
      onRetry={widget.host!.retry}
    >
      {content}
    </WidgetFailureBoundary>
  ) : (
    content
  );
};

export const WidgetHosts = () => {
  const hasWorkflowWidget = useWorkbenchSelector(({ activeLayout, activeProject }) =>
    projectNeedsWorkflowHost(activeLayout, activeProject.workflows)
  );
  const widgets = getWidgetHosts().filter((widget) => widget.manifest.id !== 'workflow' || hasWorkflowWidget);

  return (
    <>
      {widgets.map((widget) => (
        <WidgetHostBoundary key={widget.manifest.id} widget={widget} />
      ))}
    </>
  );
};
