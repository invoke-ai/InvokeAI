import { DiagnosticsPanel } from '@workbench/diagnostics/DiagnosticsPanel';
import { useActiveProjectId } from '@workbench/WorkbenchContext';

export const DiagnosticsWidgetView = () => <DiagnosticsPanel projectId={useActiveProjectId()} />;
