import { deepClone } from 'common/util/deepClone';
import { merge } from 'lodash-es';
import { ClickScrollPlugin, OverlayScrollbars } from 'overlayscrollbars';
import type { UseOverlayScrollbarsParams } from 'overlayscrollbars-react';
import type { CSSProperties } from 'react';

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

export const getOverlayScrollbarsParams = ({
  overflowX = 'hidden',
  overflowY = 'scroll',
  visibility = 'auto',
}: {
  overflowX?: 'hidden' | 'scroll';
  overflowY?: 'hidden' | 'scroll';
  visibility?: 'auto' | 'hidden' | 'visible';
}) => {
  const params = deepClone(overlayScrollbarsParams);
  merge(params, {
    options: {
      overflow: { y: overflowY, x: overflowX },
      scrollbars: { visibility, autoHide: visibility === 'visible' ? 'never' : 'scroll' },
    },
  });
  return params;
};

export const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};
