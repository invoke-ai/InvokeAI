import { Box, Flex, HStack, Icon, VisuallyHidden } from '@chakra-ui/react';
import { ensureInvocationTemplatesLoaded } from '@features/workflow/react';
import { useWorkflowHostCommands, useWorkflowProjectSelector } from '@features/workflow/ui/WorkflowUiContext';
import { useMountEffect } from '@platform/react/useMountEffect';
import { Scrollable, Tabs } from '@platform/ui';
import { ResizeHandle } from '@platform/ui/ResizeHandle';
import { EyeIcon, PencilIcon } from 'lucide-react';
import { useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';

import { FormBuilderTab } from './FormBuilderTab';
import { LinearFormView } from './LinearFormView';
import { NodeInspector } from './NodeInspector';
import { WorkflowDetailsTab } from './WorkflowDetailsTab';
import { WorkflowJsonTab } from './WorkflowJsonTab';

type PanelMode = 'view' | 'edit';
type EditTab = 'form' | 'details' | 'json';
type PanelModeItem = {
  labelKey: string;
  icon: typeof EyeIcon;
  mode: PanelMode;
};

/** Inspector share of the edit panel's height, as a percentage; the content keeps at least a quarter. */
const DEFAULT_INSPECTOR_SIZE_PCT = 35;
const MIN_INSPECTOR_SIZE_PCT = 12;
const MAX_INSPECTOR_SIZE_PCT = 75;
const PANEL_MODES: PanelModeItem[] = [
  { labelKey: 'common.view', icon: EyeIcon, mode: 'view' },
  { labelKey: 'common.edit', icon: PencilIcon, mode: 'edit' },
];

export interface WorkflowPanelState {
  editTab: EditTab;
  inspectorSizePct: number;
  mode: PanelMode;
}

const getPanelMode = (values: Record<string, unknown>): PanelMode => (values.panelMode === 'edit' ? 'edit' : 'view');

const getEditTab = (values: Record<string, unknown>): EditTab =>
  values.editTab === 'details' || values.editTab === 'json' ? values.editTab : 'form';

const getInspectorSizePct = (values: Record<string, unknown>): number =>
  typeof values.inspectorSizePct === 'number' && Number.isFinite(values.inspectorSizePct)
    ? Math.min(MAX_INSPECTOR_SIZE_PCT, Math.max(MIN_INSPECTOR_SIZE_PCT, values.inspectorSizePct))
    : DEFAULT_INSPECTOR_SIZE_PCT;

export const getWorkflowPanelState = (values: Record<string, unknown>): WorkflowPanelState => ({
  editTab: getEditTab(values),
  inspectorSizePct: getInspectorSizePct(values),
  mode: getPanelMode(values),
});

export const areWorkflowPanelStatesEqual = (left: WorkflowPanelState, right: WorkflowPanelState): boolean =>
  left.mode === right.mode && left.editTab === right.editTab && left.inspectorSizePct === right.inspectorSizePct;

/**
 * Use tabs for roving focus and arrow selection; hidden content nodes preserve aria-controls while edit panels
 * retain their splitter layout.
 */
export const PanelModeToggle = ({ mode, onChange }: { mode: PanelMode; onChange: (mode: PanelMode) => void }) => {
  const { t } = useTranslation();
  const onValueChange = useCallback(
    (event: { value: string }) => onChange(event.value === 'edit' ? 'edit' : 'view'),
    [onChange]
  );

  return (
    <Tabs.Root mb="-1" size="sm" value={mode} variant="line" onValueChange={onValueChange}>
      <Tabs.List aria-label={t('widgets.workflow.panelMode')}>
        {PANEL_MODES.map(({ labelKey, icon, mode: itemMode }) => (
          <Tabs.Trigger key={itemMode} value={itemMode}>
            <Icon as={icon} boxSize="3" />
            {t(labelKey)}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      {PANEL_MODES.map(({ labelKey, mode: itemMode }) => (
        <Tabs.Content key={itemMode} value={itemMode} asChild>
          <VisuallyHidden>{t(labelKey)}</VisuallyHidden>
        </Tabs.Content>
      ))}
    </Tabs.Root>
  );
};

export const WorkflowLinearPanel = () => {
  const { t } = useTranslation();
  const { editTab, inspectorSizePct, mode } = useWorkflowProjectSelector((project) =>
    getWorkflowPanelState(project.workflowValues)
  );
  const { widgets } = useWorkflowHostCommands();

  useMountEffect(() => {
    ensureInvocationTemplatesLoaded();
  });

  const patchValues = useCallback(
    (values: Record<string, unknown>) => widgets.patchValues('workflow', values),
    [widgets]
  );
  const onPanelModeChange = useCallback((panelMode: PanelMode) => patchValues({ panelMode }), [patchValues]);
  const onEditTabChange = useCallback(
    (event: { value: string }) => patchValues({ editTab: event.value }),
    [patchValues]
  );

  return (
    <Flex direction="column" flex="1" h="full" minH="0">
      <HStack flexShrink={0} justify="space-between" px="2" h={10} borderBottomWidth={1}>
        <PanelModeToggle mode={mode} onChange={onPanelModeChange} />
        {mode === 'edit' ? (
          <Tabs.Root size="sm" value={editTab} variant="outline" mb="-1" onValueChange={onEditTabChange}>
            <Tabs.List>
              <Tabs.Trigger value="form" fontSize="2xs">
                {t('widgets.workflow.form')}
              </Tabs.Trigger>
              <Tabs.Trigger value="details" fontSize="2xs">
                {t('widgets.workflow.details')}
              </Tabs.Trigger>
              <Tabs.Trigger value="json" fontSize="2xs">
                {t('common.json')}
              </Tabs.Trigger>
            </Tabs.List>
            <Tabs.Content value="form" asChild>
              <VisuallyHidden>{t('widgets.workflow.form')}</VisuallyHidden>
            </Tabs.Content>
            <Tabs.Content value="details" asChild>
              <VisuallyHidden>{t('widgets.workflow.details')}</VisuallyHidden>
            </Tabs.Content>
            <Tabs.Content value="json" asChild>
              <VisuallyHidden>{t('common.json')}</VisuallyHidden>
            </Tabs.Content>
          </Tabs.Root>
        ) : null}
      </HStack>
      {mode === 'view' ? (
        <WorkflowLinearViewContent />
      ) : (
        <WorkflowLinearEditContent editTab={editTab} inspectorSizePct={inspectorSizePct} patchValues={patchValues} />
      )}
    </Flex>
  );
};

const WorkflowLinearViewContent = () => {
  const { t } = useTranslation();
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);

  return (
    <Scrollable flex="1" label={t('widgets.workflow.panelContent')} minH="0">
      <LinearFormView projectGraph={projectGraph} />
    </Scrollable>
  );
};

const WorkflowLinearEditContent = ({
  editTab,
  inspectorSizePct,
  patchValues,
}: {
  editTab: EditTab;
  inspectorSizePct: number;
  patchValues: (values: Record<string, unknown>) => void;
}) => {
  const { t } = useTranslation();
  const projectGraph = useWorkflowProjectSelector((project) => project.projectGraph);
  const inspectorRef = useRef<HTMLDivElement>(null);
  const onCommitInspectorSize = useCallback(
    (inspectorSizePct: number) => patchValues({ inspectorSizePct }),
    [patchValues]
  );

  return (
    <Flex direction="column" flex="1" minH="0">
      <Box flex="1" minH="0" minW="0" overflow="hidden">
        {editTab === 'json' ? (
          <WorkflowJsonTab projectGraph={projectGraph} />
        ) : (
          <Scrollable flex="1" h="full" label={t('widgets.workflow.panelContent')} minH="0" minW="0" w="full">
            {editTab === 'form' ? (
              <FormBuilderTab projectGraph={projectGraph} />
            ) : (
              <WorkflowDetailsTab metadata={projectGraph} />
            )}
          </Scrollable>
        )}
      </Box>
      <ResizeHandle
        label={t('widgets.workflow.resizeNodeInspector')}
        max={MAX_INSPECTOR_SIZE_PCT}
        min={MIN_INSPECTOR_SIZE_PCT}
        orientation="horizontal"
        pane="after"
        paneRef={inspectorRef}
        sizeProperty="flexBasis"
        unit="%"
        value={inspectorSizePct}
        onCommit={onCommitInspectorSize}
      />
      <Box ref={inspectorRef} flex={`0 0 ${inspectorSizePct}%`} minH="0" overflow="hidden">
        <NodeInspector projectGraph={projectGraph} />
      </Box>
    </Flex>
  );
};
