import { getWidgetHosts } from '@workbench/widgetRegistry';

export const WidgetHosts = () => (
  <>
    {getWidgetHosts().map((widget) => {
      const Host = widget.manifest.host;

      return Host ? <Host key={widget.manifest.id} /> : null;
    })}
  </>
);
