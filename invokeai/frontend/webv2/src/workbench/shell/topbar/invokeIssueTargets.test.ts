import { describe, expect, it } from 'vitest';

import { getInvocationIssueTarget } from './invokeIssueTargets';

describe('getInvocationIssueTarget', () => {
  it('leads a workflow issue about a node to that node, and a graph-wide one to the workflow', () => {
    expect(
      getInvocationIssueTarget(
        { workflowIssue: { message: 'x', messageKey: 'nodes.readiness.missingInput', nodeId: 'node-7' } },
        'workflow'
      )
    ).toEqual({ kind: 'workflowNode', nodeId: 'node-7' });
    expect(
      getInvocationIssueTarget({ workflowIssue: { message: 'x', messageKey: 'nodes.readiness.cycle' } }, 'workflow')
    ).toEqual({ kind: 'widget', widgetId: 'workflow' });
  });

  it('offers to add a missing source widget, and the Gallery for video results', () => {
    expect(
      getInvocationIssueTarget({ routeIssue: { kind: 'sourceNotMounted', sourceId: 'workflow' } }, 'workflow')
    ).toEqual({
      kind: 'mountSource',
      sourceId: 'workflow',
    });
    expect(getInvocationIssueTarget({ routeIssue: { kind: 'videoNeedsGallery' } }, 'video')).toEqual({
      destination: 'gallery',
      kind: 'destination',
    });
  });

  it('leads a blocking control layer to that layer, and a hosted refusal to the Canvas', () => {
    expect(
      getInvocationIssueTarget(
        {
          canvasInputIssue: { code: 'missing_model', kind: 'control_layer', layerId: 'control-1', layerName: 'Sketch' },
        },
        'canvas'
      )
    ).toEqual({ kind: 'layer', layerId: 'control-1' });
    expect(
      getInvocationIssueTarget(
        {
          canvasInputIssue: {
            kind: 'hosted_refusal',
            modelName: 'OpenAI Image',
            refusal: { kind: 'regional_guidance', layerName: 'Hat' },
          },
        },
        'canvas'
      )
    ).toEqual({ kind: 'widget', widgetId: 'canvas' });
  });

  it('leads a plain reason to the source widget that owns it', () => {
    expect(getInvocationIssueTarget('Upscale settings are incomplete.', 'upscale')).toEqual({
      kind: 'widget',
      widgetId: 'upscale',
    });
  });
});
