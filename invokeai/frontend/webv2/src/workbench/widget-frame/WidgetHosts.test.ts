import { describe, expect, it } from 'vitest';

import { projectHasWidgetType, projectNeedsWorkflowHost } from './WidgetHosts';

const project = (instanceIds: string[], floatingIds: string[] = []) => ({
  floatingWidgets: Object.fromEntries(floatingIds.map((id) => [id, {}])),
  widgetInstances: Object.fromEntries(
    [...instanceIds, ...floatingIds].map((id) => [id, { id, typeId: id.startsWith('workflow') ? 'workflow' : 'queue' }])
  ),
  widgetRegions: {
    center: { instanceIds },
    left: { instanceIds: [] },
    right: { instanceIds: [] },
    bottom: { instanceIds: [] },
  },
});

describe('projectHasWidgetType', () => {
  it('finds widget types mounted in regions', () => {
    expect(projectHasWidgetType(project(['workflow-1']), 'workflow')).toBe(true);
    expect(projectHasWidgetType(project(['queue-1']), 'workflow')).toBe(false);
  });

  it('finds widget types mounted in floating windows', () => {
    expect(projectHasWidgetType(project([], ['workflow-1']), 'workflow')).toBe(true);
  });

  it('mounts the workflow host for a call-saved-workflow graph without a workflow widget', () => {
    expect(
      projectNeedsWorkflowHost({
        ...project([]),
        workflows: {
          entries: [{ document: { nodes: [{ data: { type: 'call_saved_workflow' }, type: 'invocation' }] } }],
        },
      })
    ).toBe(true);
  });

  it('does not mount the workflow host for unrelated graphs', () => {
    expect(
      projectNeedsWorkflowHost({
        ...project([]),
        workflows: { entries: [{ document: { nodes: [{ data: { type: 'noise' }, type: 'invocation' }] } }] },
      })
    ).toBe(false);
  });

  it('mounts the workflow host when an inactive workflow calls a saved workflow', () => {
    expect(
      projectNeedsWorkflowHost({
        ...project([]),
        workflows: {
          entries: [
            { document: { nodes: [] } },
            { document: { nodes: [{ data: { type: 'call_saved_workflow' }, type: 'invocation' }] } },
          ],
        },
      })
    ).toBe(true);
  });
});
