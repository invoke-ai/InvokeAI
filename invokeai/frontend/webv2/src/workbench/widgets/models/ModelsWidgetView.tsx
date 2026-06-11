import type { WidgetViewProps } from '../../types';
import { ModelsCenterView } from './ModelsCenterView';
import { ModelsPanelView } from './ModelsPanelView';

/**
 * Region router for the models widget: the center region gets the full
 * manager (library, add models, install queue); side panels get the compact
 * browser with drill-in detail.
 */
export const ModelsWidgetView = ({ region }: WidgetViewProps) => {
  if (region === 'center') {
    return <ModelsCenterView />;
  }

  return <ModelsPanelView />;
};
