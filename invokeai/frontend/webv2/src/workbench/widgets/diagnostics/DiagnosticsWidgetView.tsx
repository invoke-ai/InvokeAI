import type { WidgetViewProps } from '@workbench/widgetContracts';

import { DiagnosticsPanel } from '@workbench/diagnostics/DiagnosticsPanel';
import { useProblemCount } from '@workbench/diagnostics/useProblemCount';
import { StatusWidgetChip } from '@workbench/widget-frame';
import { useActiveProjectId } from '@workbench/WorkbenchContext';
import { BugIcon, ClipboardListIcon } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const DiagnosticsWidgetView = ({ presentation, region }: WidgetViewProps) => {
  const { t } = useTranslation();
  const projectId = useActiveProjectId();
  const problemCount = useProblemCount(projectId);

  if (region === 'bottom' && presentation !== 'expanded') {
    const label =
      problemCount === 0
        ? t('widgets.diagnostics.chipClean')
        : t('widgets.diagnostics.chipProblems', { count: problemCount });

    return (
      <StatusWidgetChip icon={problemCount > 0 ? BugIcon : ClipboardListIcon}>
        {t('widgets.diagnostics.chipLabel', { label })}
      </StatusWidgetChip>
    );
  }

  return <DiagnosticsPanel projectId={projectId} />;
};
