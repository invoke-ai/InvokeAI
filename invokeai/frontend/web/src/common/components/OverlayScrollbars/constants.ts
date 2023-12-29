import { ClickScrollPlugin, OverlayScrollbars } from 'overlayscrollbars';
import type { UseOverlayScrollbarsParams } from 'overlayscrollbars-react';

OverlayScrollbars.plugin(ClickScrollPlugin);

export const overlayScrollbarsParams: UseOverlayScrollbarsParams = {
  defer: true,
  options: {
    scrollbars: {
      visibility: 'auto',
      autoHide: 'scroll',
      autoHideDelay: 1300,
      theme: 'os-theme-dark',
      clickScroll: true,
    },
    overflow: { x: 'hidden' },
  },
};
