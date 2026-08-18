import type { GenerateWidgetValues } from '@features/generation/contracts';
import type { ModelConfig } from '@features/models';
import type {
  GraphPreviewNotice,
  GraphPreviewProvenance,
  GraphPreviewSourceState,
  GraphPreviewSummaryRow,
} from '@features/workflow/contracts';
import type { InvocationTemplatesSnapshot } from '@features/workflow/react';
import type { Project } from '@workbench/projectContracts';
import type { GraphBearingSurfaceContract } from '@workbench/widgetContracts';
import type { TFunction } from 'i18next';

import { compileGeneratePreviewGraph, getGenerateNodeProvenance } from '@features/generation/graph';
import { compileProjectGraph } from '@features/workflow/graph';
import { getProjectWidgetValues } from '@workbench/widgetState';

/**
 * Pure translation from the active project + surface into the preview
 * dialog's data. Kept side-effect free so it can run on every keystroke
 * (`GraphPreviewHost` recomputes it in a `useMemo`) without triggering any
 * loads itself — callers are responsible for ensuring models/templates are
 * fetched (`ensureModelsLoaded`, `ensureInvocationTemplatesLoaded`).
 */
export interface GraphPreviewSourceDeps {
  models: readonly ModelConfig[] | undefined;
  project: Project;
  surface: GraphBearingSurfaceContract;
  t: TFunction;
  templates: InvocationTemplatesSnapshot;
}

const buildGenerateSummaryRows = (settings: GenerateWidgetValues, t: TFunction): GraphPreviewSummaryRow[] => {
  const activeLoras = settings.loras.filter((lora) => lora.isEnabled).length;

  return [
    { id: 'model', label: t('graphPreview.model'), value: settings.model.name },
    { id: 'size', label: t('graphPreview.size'), value: `${settings.width} × ${settings.height}` },
    { id: 'steps', label: t('graphPreview.steps'), value: String(settings.steps) },
    { id: 'cfgScale', label: t('graphPreview.cfgScale'), value: String(settings.cfgScale) },
    { id: 'scheduler', label: t('graphPreview.scheduler'), value: settings.scheduler },
    ...(activeLoras > 0 ? [{ id: 'loras', label: t('graphPreview.loras'), value: String(activeLoras) }] : []),
    {
      id: 'seed',
      label: t('graphPreview.seed'),
      value: settings.shouldRandomizeSeed ? t('graphPreview.seedRandomValue') : String(settings.seed),
    },
  ];
};

const EMPTY_SOURCE_BASE: Pick<GraphPreviewSourceState, 'invalidReasons' | 'notices' | 'summaryRows'> = {
  invalidReasons: [],
  notices: [],
  summaryRows: [],
};

const buildWorkflowSource = (project: Project, templates: InvocationTemplatesSnapshot): GraphPreviewSourceState => {
  if (templates.status !== 'loaded') {
    return { ...EMPTY_SOURCE_BASE, graph: null, isLive: true };
  }

  const positionHints = Object.fromEntries(project.projectGraph.nodes.map((node) => [node.id, node.position]));

  try {
    const graph = compileProjectGraph(project.projectGraph, templates.templates);

    return { ...EMPTY_SOURCE_BASE, graph, isLive: true, positionHints };
  } catch {
    return { ...EMPTY_SOURCE_BASE, graph: null, isLive: true, positionHints };
  }
};

const buildGenerateSource = (
  project: Project,
  models: readonly ModelConfig[] | undefined,
  t: TFunction
): GraphPreviewSourceState => {
  const result = compileGeneratePreviewGraph({
    destination: project.invocation.destination,
    models: models ?? [],
    storedValues: getProjectWidgetValues(project, 'generate'),
    useCpuNoise: project.settings.useCpuNoise,
  });

  if (result.status === 'invalid') {
    return { ...EMPTY_SOURCE_BASE, graph: null, invalidReasons: result.reasons, isLive: true };
  }

  const { settings } = result;
  const isSeedRandomized = settings.shouldRandomizeSeed;
  const notices: GraphPreviewNotice[] = isSeedRandomized
    ? [{ id: 'seed-random', message: t('graphPreview.seedRandomized'), nodeId: 'seed' }]
    : [];
  const resolvedInputOverrides = isSeedRandomized ? { seed: { value: t('graphPreview.seedRegenerated') } } : undefined;
  const getProvenance = (nodeId: string, fieldName: string): GraphPreviewProvenance | null => {
    const entry = getGenerateNodeProvenance(nodeId, fieldName);

    if (!entry) {
      return null;
    }

    if (entry.settingKey === 'seed' && isSeedRandomized) {
      return { label: t('graphPreview.provenance.seedRandom') };
    }

    return { label: t(entry.labelKey) };
  };

  return {
    getProvenance,
    graph: result.graph,
    invalidReasons: [],
    isLive: true,
    notices,
    resolvedInputOverrides,
    summaryRows: buildGenerateSummaryRows(settings, t),
  };
};

const buildWidgetGraphSource = (project: Project, surface: GraphBearingSurfaceContract): GraphPreviewSourceState => ({
  ...EMPTY_SOURCE_BASE,
  graph: project.widgetGraphs[surface.widgetId] ?? null,
  isLive: false,
});

export const buildGraphPreviewSource = ({
  models,
  project,
  surface,
  t,
  templates,
}: GraphPreviewSourceDeps): GraphPreviewSourceState => {
  switch (surface.sourceId) {
    case 'workflow':
      return buildWorkflowSource(project, templates);
    case 'generate':
      return buildGenerateSource(project, models, t);
    case 'upscale':
    case 'canvas':
      return buildWidgetGraphSource(project, surface);
  }
};
